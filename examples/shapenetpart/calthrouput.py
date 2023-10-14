"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import argparse
import yaml
import os
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
from Point_cloud_postprocessing import pointscalarea,pointcaldistance


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

def cal_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

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
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,)

    from openpoints.models.pointmlp.pointMLP import pointMLP
    from openpoints.models.curvenet.curvenet_seg import CurveNet
    from openpoints.models.dgcnn.model import DGCNN_partseg
    from openpoints.models.pct.pct import Point_Transformer_partseg
    from openpoints.models.pointnet2.pointnet2_part_seg_msg import pointnet2

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).cuda()




    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   #  from openpoints.models.pointmlp.pointMLP import pointMLP
   #  from openpoints.models.curvenet.curvenet_seg import CurveNet
  #  model = pointMLP(3, 2048).to(device)  # partnums, points
    #model = DGCNN_partseg().cuda()
    #model = Point_Transformer_partseg().cuda()
   # model = CurveNet().to(device)
    #model  = pointnet2().cuda()

    import numpy as np

    optimal_batch_size  = 1
    pos = torch.randn(optimal_batch_size, 2048, 3)
    pos = pos.to(device)
    #print(type(pos))
    from thop import clever_format
    xvalue = torch.rand(optimal_batch_size, 7, 2048)
    #print(xvalue)
    xvalue = xvalue.to(device)

    yvalue = np.array([optimal_batch_size, 2048], dtype=np.int64)
    yvalue = torch.from_numpy(yvalue)
    #yvalue = torch.zeros(1, 2048)   #torch.randn(1,2048)
    yvalue = yvalue.to(device)

    cls = np.array([[0]*optimal_batch_size], dtype=np.int64).reshape((optimal_batch_size,1))
    cls = torch.from_numpy(cls)
    cls = cls.to(device)

    from thop import profile
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    data = {'pos': pos,
            'x': xvalue,
            'y': yvalue,
            'cls': cls}
    model.eval()

    label_one_hot = np.zeros((data['cls'].shape[0], 1))
    for idx in range(data['cls'].shape[0]):
        label_one_hot[idx, data['cls'][idx]] = 1
    label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
    label_one_hot = label_one_hot.cuda()

    ################# infer speed
    for _ in range(50):
        _ = model(data)

    iterations = 300
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(data)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            #print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    ##########################

    ################through put


    # repetitions = 100
    # total_time = 0
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #         starter.record()
    #         _ = model(data)
    #         ender.record()
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender) / 1000
    #         total_time += curr_time
    # Throughput = (repetitions * optimal_batch_size) / total_time
    # print("FinalThroughput:", Throughput)








    ##########################









    wandb.finish(exit_code=True)






















if __name__ == "__main__":
    parser = argparse.ArgumentParser('ShapeNetPart Part segmentation training')         #pointnextmydatacfgs
    parser.add_argument('--cfg', type=str, default="D:/pythonproject/Pointnext/cfgs/shapenetpart/pointnextmydatacfgs.yaml", help='config file')
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
