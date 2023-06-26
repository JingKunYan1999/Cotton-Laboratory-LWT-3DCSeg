import os
import glob
import h5py
import json
import pickle
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from openpoints.models.layers import fps, furthest_point_sample
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from openpoints.transforms import build_transforms_from_cfg



def load_data_partseg(partition, DATA_DIR):
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, '*train*.h5')) \
            + glob.glob(os.path.join(DATA_DIR, '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, '*%s*.h5' % partition))
    for h5_name in file:
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(
        pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix)  # random rotation (x,z)
    return pointcloud

class ShapeNetPartNormal(Dataset):
    #大类
    # classes = ['airplane', 'bag', 'cap', 'car', 'chair',
    #            'earphone', 'guitar', 'knife', 'lamp', 'laptop',
    #            'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    classes = ['cotton']


    #大类部件对应关系
    # cls_parts = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
    #                'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37],
    #                'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
    #                'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40], 'chair': [12, 13, 14, 15], 'knife': [22, 23]}
    cls_parts = {'cotton': [0, 1, 2]}


    cls2parts = []
    cls2partembed = torch.zeros(1, 3)
    # cls2partembed = torch.zeros(16, 50)


    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat

    def __init__(self,
                 data_root='data/useddatasetxyz',   #'data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 num_points=2048,
                 class_choice=None,
                 use_normal=True,
                 shape_classes=1, #16
                 presample=False,
                 sampler='fps',
                 split='test',
                 transform=None,
                 multihead=False,
                 ):
        self.npoints = num_points
        self.root = data_root
        self.transform = transform
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.use_normal = use_normal
        self.presample = presample
        self.sampler = sampler 
        self.split = split
        self.multihead=multihead
        self.part_start = [0]
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            fns = [fn for fn in fns if fn[0:-4] in test_ids]
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        if transform is None:
            self.eye = np.eye(shape_classes)
        else:
            self.eye = torch.eye(shape_classes)

        # in the testing, using the uniform sampled 2048 points as input
        # presample
        filename = os.path.join(data_root, 'processed',
                                f'{split}_{num_points}_fps.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data, self.cls = [], []
            npoints = []
            for cat, filepath in tqdm(self.datapath, desc=f'Sample ShapeNetPart {split} split'):
                cls = self.classes[cat]
                cls = np.array([cls]).astype(np.int64)
                data = np.loadtxt(filepath).astype(np.float32)
                npoints.append(len(data))
                data = torch.from_numpy(data).to(
                    torch.float32).cuda().unsqueeze(0)
                data = fps(data, num_points).cpu().numpy()[0]
                self.data.append(data)
                self.cls.append(cls)
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                'test', np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(os.path.join(data_root, 'processed'), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump((self.data, self.cls), f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data, self.cls = pickle.load(f)
                print(f"{filename} load successfully")

    def __getitem__(self, index):
        if not self.presample:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int64)
        else:
            data, cls = self.data[index], self.cls[index]
            point_set, seg = data[:, :6], data[:, 6].astype(np.int64)


        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice]
        seg = seg[choice]
        ori_data = point_set
        if self.multihead:
            seg=seg-self.part_start[cls[0]]

        data = {'pos': point_set[:, 0:3],
                'x': point_set[:, 3:6],
                'cls': cls,
                'y': seg,
                }
        data_transform = build_transforms_from_cfg("val", self.transform)
        data = data_transform(data)
        data['ori_data'] = ori_data
        return data

    def __len__(self):
        return len(self.datapath)



if __name__ == '__main__':
    train = ShapeNetPartNormal(num_points=2048, split='trainval')
    test = ShapeNetPartNormal(num_points=2048, split='test')
    for dict in train:
        for i in dict:
            print(i, dict[i].shape)