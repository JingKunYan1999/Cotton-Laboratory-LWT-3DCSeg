"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import numpy as np
import random
import time
import argparse
import calendar
import yaml
import os
import open3d as o3d
import sys
import logging
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import warnings
import numpy as np
#from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter
from datautils.forafterpointDataLoader import ShapeNetPartNormal
from Point_cloud_postprocessing import pointscalarea,pointcaldistance,newcallengandwidth,calculatestem,savepredictedply,leafinstancesegment
import shutil



torch.backends.cudnn.benchmark = False
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.models import build_model_from_cfg
from openpoints.models.layers import torch_grouping_operation, knn_point
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_class_weights, get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.models.layers import furthest_point_sample



def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


def get_ins_mious(pred, target, cls, cls2parts,
                  multihead=False,
                  ):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    final_part_ious = []

    batchpart_iou = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        mypart_iou = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)

            mypart_iou.append(iou.cpu().float())  #自加


        batchpart_iou.append(mypart_iou)


        ins_mious.append(torch.mean(torch.stack(part_ious)))

    batchpart_iou = np.array(batchpart_iou)
    batchpart_iou = np.mean(batchpart_iou,axis=0)
    #print(f"batchpart_iou{batchpart_iou}")




    return ins_mious,batchpart_iou


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        #writer = SummaryWriter(log_dir=cfg.run_dir)
        writer = None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # build dataset



    root = 'D:/pythonproject/Pointnext/data/ShapeNetPart/typeb_120_useddatasetxyz'
    numpoints = 2048
    batch_size = 8
    TEST_DATASET = ShapeNetPartNormal(data_root=root, num_points=numpoints, split='test',transform=cfg.datatransforms)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=4)
    cfg.cls2parts = testDataLoader.dataset.cls2parts



    validate_fn = eval(cfg.get('val_fn', 'validate'))

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).cuda()
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')


    # transforms
    if 'vote' in cfg.datatransforms:
        voting_transform = build_transforms_from_cfg('vote', cfg.datatransforms)
    else:
        voting_transform = None



    pretrained_path = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\typeb_120_83.78\checkpoint\ckpt_best.pth"
    load_checkpoint(model, pretrained_path=pretrained_path)
    validate_fn(model, testDataLoader, cfg,
                 num_votes=cfg.num_votes,
                data_transform=voting_transform)



    wandb.finish(exit_code=True)




@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=0, data_transform=None):
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []
    part_iou = []

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:

        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        batch_size, num_point, _ = data['pos'].size()
        logits = 0
        for v in range(num_votes+1):
            set_random_seed(v)
            if v > 0:
                data['pos'] = data_transform(data['pos'])
            logits += model(data)
        logits /= (num_votes + 1)
        preds = logits.max(dim=1)[1]   #preds即为所求类别
        #print("preds",preds.shape)
        #print(preds)
        if cfg.get('refine', False):
            part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 10))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious,batch_part_iou = get_ins_mious(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious
            part_iou.append(batch_part_iou)
        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_ins_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts,
                                    multihead=True)
                iou_array.append(iou)
            ins_miou_list += iou_array

        # per category iou at each batch_size:
        for shape_idx in range(batch_size):  # sample_idx
            cur_gt_label = int(cls[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    if cfg.distributed:
        dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

    for cat_idx in range(cfg.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    part_iou = np.array(part_iou)
    part_iou = np.mean(part_iou, axis=0)
    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                        f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'part IoUs {part_iou},' 
                        f'\n Class mIoUs {cls_mious}')
    return ins_miou, cls_miou, part_iou, cls_mious


if __name__ == "__main__":

    parser = argparse.ArgumentParser('ShapeNetPart Part segmentation training')
    parser.add_argument('--cfg', type=str, default="D:/pythonproject/Pointnext/cfgs/shapenetpart/mydatacfgs.yaml", help='config file')
    #args.cfg = "D:/pythonproject/Pointnext/cfgs/shapenetpart/default.yaml"
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # logger
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    if cfg.mode in ['resume', 'test', 'val']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name


    main(0, cfg)
