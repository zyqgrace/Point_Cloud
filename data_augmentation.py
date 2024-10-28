import argparse
import numpy as np
import pprint
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import logging
from progressbar import ProgressBar
from tqdm import tqdm
from dataloader import create_dataloader
from time import time
from datetime import datetime
import sys
from collections import defaultdict
from configs import get_cfg_defaults
from pointnet_pyt.pointnet.model import feature_transform_regularizer
import random
import models
import aug_utils
from third_party import bn_helper, tent_helper
import importlib
from all_utils import (
    TensorboardManager, PerfTrackTrain,
    PerfTrackVal, TrackTrain, smooth_loss, DATASET_NUM_CLASS)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_metric_from_perf(task, perf, metric_name):
    if task in ['cls', 'cls_trans']:
        assert metric_name in ['acc']
        metric = perf[metric_name]
    else:
        assert False
    return metric

def adapt_bn(data,model,cfg):
    model = bn_helper.configure_model(model,eps=1e-5, momentum=0.1,reset_stats=False,no_stats=False)
    for _ in range(cfg.ITER):
        model(**data) 
    print("Adaptation Done ...")
    model.eval()
    return model

def adapt_tent(data,model,cfg):
    model = tent_helper.configure_model(model,eps=1e-5, momentum=0.1)
    parameter,_ = tent_helper.collect_params(model)
    optimizer_tent = torch.optim.SGD(parameter, lr=0.001,momentum=0.9)

    for _ in range(cfg.ITER):
        # index = np.random.choice(args.number,args.batch_size,replace=False)
        tent_helper.forward_and_adapt(data,model,optimizer_tent)
    print("Adaptation Done ...")
    model.eval()
    return model


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--entry', type=str, default="train")
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--model-path', type=str, default="")
    parser.add_argument('--exp-config', type=str, default="")
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default="log", help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--corruption',type=str,default='uniform', help="Which corruption to use")
    parser.add_argument('--output',type=str,default='./test.txt', help="path to output file")
    parser.add_argument('--severity',type=int,default=1, help="Which severity to use")
    parser.add_argument('--confusion', action="store_true", default=False, help="whether to output confusion matrix data")
    return parser.parse_args()

    

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cpu':
        print('WARNING: Using CPU')
        
    '''CREATE DIR'''
    experiment_dir = 'run/detect/' + args.log_dir
    
    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/ModelNet40'
        
    assert not args.exp_config == ""
    if not args.resume:
        assert args.model_path == ""
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.exp_config)
    if cfg.EXP.EXP_ID == "":
            cfg.EXP.EXP_ID = str(datetime.now())[:-7].replace(' ', '-')
    cfg.freeze()
    
    random.seed(cfg.EXP.SEED)
    np.random.seed(cfg.EXP.SEED)
    torch.manual_seed(cfg.EXP.SEED)

    SAVE_DIR = "data/modelnet_no_aug"
    
    loader_train = create_dataloader(split='train', cfg=cfg)
    
    augmentation_name = ''
    print(f"Applying augmentation: {augmentation_name}")
    for i, data_batch in enumerate(loader_train):
        # Apply the chosen augmentation
        if augmentation_name == 'cutmix_r':
            data_batch = aug_utils.cutmix_r(data_batch, cfg)
        elif augmentation_name == 'cutmix_k':
            data_batch = aug_utils.cutmix_k(data_batch, cfg)
        elif augmentation_name == 'mixup':
            data_batch = aug_utils.mixup(data_batch, cfg)
        elif augmentation_name == 'rsmix':
            data_batch = aug_utils.rsmix(data_batch, cfg)
        elif augmentation_name == 'pgd':
            data_batch = aug_utils.pgd(data_batch, model=None, task=None, loss_name=None, dataset_name=None)
            
        # Save the augmented batch to the SAVE_DIR
        save_path = os.path.join(SAVE_DIR, f"augmented_batch_{i}.pt")
        torch.save(data_batch, save_path)
        print(f"Saved augmented data batch {i} to {save_path}")

    print("Data augmentation and saving complete.")


if __name__ == '__main__':
    args = parse_args()
    main(args)