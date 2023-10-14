"""
Distributed training script for scene segmentation with S3DIS dataset
"""
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
from datautils.tempforafterpointDataLoader import ShapeNetPartNormal
from Point_cloud_postprocessing import pointscalarea,pointcaldistance,newcallengandwidth,calculatestem,savepredictedply,leafinstancesegment,savepredictedtxt
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
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
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
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious


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



    root = 'D:/pythonproject/Pointnext/data/ShapeNetPart/tomato_tested_used'
    numpoints = 32768
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



    # pretrained_path = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\sec200_89.57\checkpoint\ckpt_best.pth"
    # load_checkpoint(model, pretrained_path=pretrained_path)
    # validate_fn(model, testDataLoader, cfg,
    #              num_votes=cfg.num_votes,
    #             data_transform=voting_transform)


    ############1#############
    pretrained_path1 = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\typeb_40_72.36\checkpoint\ckpt_best.pth"
    load_checkpoint(model, pretrained_path=pretrained_path1)
    pathname1 = "40sets_"
    validate_fn(pathname1, model, testDataLoader, cfg,
                num_votes=cfg.num_votes,
                data_transform=voting_transform)
    ###########################


    ############2#############
    pretrained_path2 = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\typeb_80_78.44\checkpoint\ckpt_best.pth"
    load_checkpoint(model, pretrained_path=pretrained_path2)
    pathname2 = "80sets_"
    validate_fn(pathname2,model, testDataLoader, cfg,
                num_votes=cfg.num_votes,
                data_transform=voting_transform)
    ###########################


    ############3#############
    pretrained_path3 = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\typeb_120_83.78\checkpoint\ckpt_best.pth"
    load_checkpoint(model, pretrained_path=pretrained_path3)
    pathname3 = "120sets_"
    validate_fn(pathname3, model, testDataLoader, cfg,
                num_votes=cfg.num_votes,
                data_transform=voting_transform)
    ###########################

    ############4#############
    pretrained_path4 = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\sec200_89.57\checkpoint\ckpt_best.pth"
    load_checkpoint(model, pretrained_path=pretrained_path4)
    pathname4 = "240sets_"
    validate_fn(pathname4, model, testDataLoader, cfg,
                num_votes=cfg.num_votes,
                data_transform=voting_transform)
    ##########################





    wandb.finish(exit_code=True)




@torch.no_grad()
def validate(pathname, model, val_loader, cfg, num_votes=0, data_transform=None):
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']

        cls = data['cls']
        data['x'] = get_features_by_keys(data,cfg.feature_keys)
        batch_size, num_point, _ = data['pos'].size()
        ori_data = data['ori_data']

        newtarget = np.expand_dims(target.cpu().numpy().copy(), axis=2)  # 扩展维度 (B,N)->(B,N,1)

        sampleoritxtarr = np.concatenate((ori_data.cpu().numpy().copy(),newtarget),axis=2)

        logits = 0

        for v in range(num_votes+1):
            set_random_seed(v)
            if v > 0:
                data['pos'] = data_transform(data['pos'])
            logits += model(data)
        logits /= (num_votes + 1)
        preds = logits.max(dim=1)[1]   #preds即为所求类别

        if cfg.get('refine', False):
            part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 10))

        cur_pred_val = np.expand_dims(preds.cpu().numpy(), axis=2)  # 扩展维度 (B,N)->(B,N,1)
        seg_points = np.concatenate((ori_data.cpu().numpy(), cur_pred_val), axis=2)  # 拼接后得到有分割标签的原始数据，进行分类保存，
        bsize, _, _ = seg_points.shape
        leafarr = []
        disarr = []

        predictedpath = "D:/pythonproject/Pointnext/examples/shapenetpart/predictedpoints/"
        if not os.path.exists(predictedpath):
            os.makedirs(predictedpath)
        for item in range(bsize):  # 对batch中每个点云保存
            print()
            print(f"第{item+1}植株表型参数如下：")
            individualpath = predictedpath + str(item + 1) + "/"
            if not os.path.exists(individualpath):
                os.makedirs(individualpath)
            singlepoints = seg_points[item, :, :]  # 单个点云数据

            singlesampleoritxt = sampleoritxtarr[item, :, :]  # 单个原始数据

            savepredictedtxt(singlepoints,predictedpath + pathname + str(item+1) + ".txt") #保存可视化结果txt
            savepredictedtxt(singlesampleoritxt, predictedpath + pathname + "ori_" +str(item + 1) + ".txt")  # 保存原始可视化txt


           #  leafpoints = singlepoints[singlepoints[:, -1] == 0]  # 0代表叶子
           #  leafpoints = leafpoints[:,:-1]
           # # print(len(leafpoints))
           #  leavesarr,leafnum = leafinstancesegment(leafpoints)
           #  print(f"叶片总数量：{leafnum}")
           #  leavessavepath = individualpath + "leaves/"
           #  if not os.path.exists(leavessavepath):
           #      os.makedirs(leavessavepath)
           #  for index,leafitem in enumerate(leavesarr):  # 每一片叶子分割保存，计算叶长叶宽
           #      instanceleafpath = leavessavepath + str(index + 1) + "_leaf.ply"
           #      leaflens, leafwidths = newcallengandwidth(leafitem)  # 叶长，叶宽
           #      print(f"第{index+1}片叶子的叶长为：{leaflens/10}CM，叶宽为：{leafwidths/10}CM")
           #      o3d.io.write_point_cloud(instanceleafpath, leafitem)
           #
           #
           #  leafpath = individualpath + "leaf.ply"
           #  savepredictedply(leafpoints, leafpath)
           #
           #  stempoints = singlepoints[singlepoints[:, -1] == 1]  # 1代表茎
           #  stempoints = stempoints[:, :-1]
           #  stempath = individualpath + "stem.ply"
           #  savepredictedply(stempoints, stempath)
           #
           #
           #  stems = calculatestem(stempoints) # 茎长
           #  area = pointscalarea(leafpoints, path=None)  # 总叶面积
           #  dis = pointcaldistance(stempoints, path=None)  # 茎长
           #
           # # print(lens,widths,stems,area,dis)
           #
           #  print(f"茎直径: {stems/10}CM, 茎长: {dis/10}CM, 总叶面积: {area/100}CM2")


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
