import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet2_tf.modelnet_h5_dataset import ModelNetH5Dataset as pointnet2_ModelNetH5Dataset
from torchvision import transforms
from dgcnn.pytorch.data import ModelNet40 as dgcnn_ModelNet40
import os
import pointnet2_ops as pointnet2_utils
import rscnn.data_utils as rscnn_d_utils
from rscnn.ModelNet40Loader import ModelNet40Cls as rscnn_ModelNet40Cls


def load_data(data_path,corruption,severity):

    DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc
    
class ModelNet40PN2(Dataset):
    def __init__(self, split, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.split = split
        self.dataset_name = 'modelnet40_pn2'
        data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]
        pointnet2_params = {
            'list_filename': data_path,
            # this has nothing to do with actual dataloader batch size
            'batch_size': 32,
            'npoints': num_points,
            'shuffle': False
        }

        # loading all the pointnet2data
        self._dataset = pointnet2_ModelNetH5Dataset(**pointnet2_params)
        all_pc = []
        all_label = []
        while self._dataset.has_next_batch():
            # augmentation here has nothing to do with actual data_augmentation
            pc, label = self._dataset.next_batch(augment=False)
            all_pc.append(pc)
            all_label.append(label)
        self.all_pc = np.concatenate(all_pc)
        self.all_label = np.concatenate(all_label)

    def __len__(self):
        return self.all_pc.shape[0]

    def __getitem__(self, idx):
        return {'pc': self.all_pc[idx], 'label': np.int64(self.all_label[idx])}

    def batch_proc(self, data_batch, device):
        if self.split == "train":
            point = np.array(data_batch['pc'])
            point = self._dataset._augment_batch_data(point)
            # converted to tensor to maintain compatibility with the other code
            data_batch['pc'] = torch.tensor(point)
        else:
            pass

        return data_batch
    
class ModelNet40C(Dataset):
    def __init__(self, split, test_data_path,corruption,severity):
        assert split == 'test'
        self.split = split
        self.data_path = {
            "test":  test_data_path
        }[self.split]
        self.corruption = corruption
        self.severity = severity

        self.data, self.label = load_data(self.data_path, self.corruption, self.severity)
        # self.num_points = num_points
        self.partition =  'test'

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        return {'pc': pointcloud, 'label': label.item()}

    def __len__(self):
        return self.data.shape[0]

class ModelNet40Dgcnn(Dataset):
    def __init__(self, split, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.split = split
        self.data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]

        dgcnn_params = {
            'partition': 'train' if split in ['train', 'valid'] else 'test',
            'num_points': num_points,
            "data_path":  self.data_path
        }
        self.dataset = dgcnn_ModelNet40(**dgcnn_params)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        pc, label = self.dataset.__getitem__(idx)
        return {'pc': pc, 'label': label.item()}

# distilled from the following sources:
# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/data/ModelNet40Loader.py
# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/train_cls.py
class ModelNet40Rscnn(Dataset):
    def __init__(self, split, data_path, train_data_path,
                 valid_data_path, test_data_path, num_points):

        self.split = split
        self.num_points = num_points
        _transforms = transforms.Compose([rscnn_d_utils.PointcloudToTensor()])
        rscnn_params = {
            'num_points': 1024,  # although it does not matter
            'root': data_path,
            'transforms': _transforms,
            'train': (split in ["train", "valid"]),
            'data_file': {
                'train': train_data_path,
                'valid': valid_data_path,
                'test':  test_data_path
            }[self.split]
        }
        self.rscnn_dataset = rscnn_ModelNet40Cls(**rscnn_params)
        self.PointcloudScaleAndTranslate = PointcloudScaleAndTranslate()

    def __len__(self):
        return self.rscnn_dataset.__len__()

    def __getitem__(self, idx):
        point, label = self.rscnn_dataset.__getitem__(idx)
        # for compatibility with the overall code
        point = np.array(point)
        label = label[0].item()

        return {'pc': point, 'label': label}

    def batch_proc(self, data_batch, device):
        point = data_batch['pc'].to(device)
        if self.split == "train":
            # (B, npoint)
            fps_idx = pointnet2_utils.furthest_point_sample(point, 1200)
            fps_idx = fps_idx[:, np.random.choice(1200, self.num_points,
                                                  False)]
            point = pointnet2_utils.gather_operation(
                point.transpose(1, 2).contiguous(),
                fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            point.data = self.PointcloudScaleAndTranslate(point.data)
        else:
            fps_idx = pointnet2_utils.furthest_point_sample(
                point, self.num_points)  # (B, npoint)
            point = pointnet2_utils.gather_operation(
                point.transpose(1, 2).contiguous(),
                fps_idx).transpose(1, 2).contiguous()
        # to maintain compatibility
        point = point.cpu()
        return {'pc': point, 'label': data_batch['label']}
    
def create_dataloader(split, cfg):
    num_workers = cfg.DATALOADER.num_workers
    batch_size = cfg.DATALOADER.batch_size
    dataset_args = {
        "split": split
    }

    if cfg.EXP.DATASET == "modelnet40_c":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_C))
        dataset = ModelNet40C(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_pn2":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_PN2))
        dataset = ModelNet40PN2(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_dgcnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_DGCNN))
        dataset = ModelNet40Dgcnn(**dataset_args)
    elif cfg.EXP.DATASET == "modelnet40_rscnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_RSCNN))
        # augmentation directly done in the code so that
        # it is as similar to the vanilla code as possible
        dataset = ModelNet40Rscnn(**dataset_args)
    else:
        assert False

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = None

    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        pin_memory=(torch.cuda.is_available()) and (not num_workers)
    )
