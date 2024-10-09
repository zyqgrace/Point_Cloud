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

def train(task, loader, model, optimizer, loss_name, dataset_name, cfg):
    model.train()

    def get_extra_param():
       return None
   
    perf = PerfTrackTrain(task, extra_param=get_extra_param())
    time_forward = 0
    time_backward = 0
    time_data_loading = 0
    time3  = time()
    for i, data_batch in enumerate(loader):
        time1 = time()

        if cfg.AUG.NAME == 'cutmix_r':
            data_batch = aug_utils.cutmix_r(data_batch,cfg)
        elif cfg.AUG.NAME == 'cutmix_k':
            data_batch = aug_utils.cutmix_k(data_batch,cfg)
        elif cfg.AUG.NAME == 'mixup':
            data_batch = aug_utils.mixup(data_batch,cfg)
        elif cfg.AUG.NAME == 'rsmix':
            data_batch = aug_utils.rsmix(data_batch,cfg)
        elif cfg.AUG.NAME == 'pgd':
            data_batch = aug_utils.pgd(data_batch,model, task, loss_name, dataset_name)
            model.train()
        # print(data_batch)
        inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name)
        out = model(**inp)
        loss = get_loss(task, loss_name, data_batch, out, dataset_name)

        perf.update_all(data_batch=data_batch, out=out, loss=loss)
        time2 = time()

        if loss.ne(loss).any():
            print("WARNING: avoiding step as nan in the loss")
        else:
            optimizer.zero_grad()
            loss.backward()
            bad_grad = False
            for x in model.parameters():
                if x.grad is not None:
                    if x.grad.ne(x.grad).any():
                        print("WARNING: nan in a gradient")
                        bad_grad = True
                        break
                    if ((x.grad == float('inf')) | (x.grad == float('-inf'))).any():
                        print("WARNING: inf in a gradient")
                        bad_grad = True
                        break

            if bad_grad:
                print("WARNING: avoiding step as bad gradient")
            else:
                optimizer.step()

        time_data_loading += (time1 - time3)
        time_forward += (time2 - time1)
        time3 = time()
        time_backward += (time3 - time2)

        if i % 50 == 0:
            print(
                f"[{i}/{len(loader)}] avg_loss: {perf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")

    return perf.agg(), perf.agg_loss()

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--entry', type=str, default="test")
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

def check_inp_fmt(task, data_batch, dataset_name):
    if task in ['cls', 'cls_trans']:
        # assert set(data_batch.keys()) == {'pc', 'label'}
        # print(data_batch['pc'],data_batch['label'])
        pc, label = data_batch['pc'], data_batch['label']
        # special case made for modelnet40_dgcnn to match the
        # original implementation
        # dgcnn loads torch.DoubleTensor for the test dataset
        if dataset_name == 'modelnet40_dgcnn':
            assert isinstance(pc, torch.FloatTensor) or isinstance(
                pc, torch.DoubleTensor)
        else:
            assert isinstance(pc, torch.FloatTensor)
        assert isinstance(label, torch.LongTensor)
        assert len(pc.shape) == 3
        assert len(label.shape) == 1
        b1, _, y = pc.shape[0], pc.shape[1], pc.shape[2]
        b2 = label.shape[0]
        assert b1 == b2
        assert y == 3
        assert label.max().item() < DATASET_NUM_CLASS[dataset_name]
        assert label.min().item() >= 0
    else:
        assert NotImplemented
        
def get_inp(task, model, data_batch, batch_proc, dataset_name):
    check_inp_fmt(task, data_batch, dataset_name)
    if not batch_proc is None:
        data_batch = batch_proc(data_batch, DEVICE)
        check_inp_fmt(task, data_batch, dataset_name)

    if isinstance(model, nn.DataParallel):
        model_type = type(model.module)
    else:
        model_type = type(model)

    if task in ['cls', 'cls_trans']:
        pc = data_batch['pc']
        inp = {'pc': pc}
    else:
        assert False
    return  inp


def validate(task, loader, model, dataset_name, adapt = None, confusion = False):
    model.eval()

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0

    with torch.no_grad():
        bar = ProgressBar(max_value=len(loader))
        time5  = time()
        if confusion:
            pred = []
            ground = []
        for i, data_batch in enumerate(loader):
            time1 = time()
            inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name)
            time2 = time()

            if adapt.METHOD == 'bn':
                model = adapt_bn(inp,model,adapt)
            elif adapt.METHOD == 'tent':
                model = adapt_tent(inp,model,adapt)

            out = model(**inp)

            if confusion:
                pred.append(out['logit'].squeeze().cpu())
                ground.append(data_batch['label'].squeeze().cpu())

            time3 = time()
            perf.update(data_batch=data_batch, out=out)
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()
            bar.update(i)
            
    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    if not confusion:
        return perf.agg()
    else:
        pred = np.argmax(torch.cat(pred).numpy(), axis=1)
        # print(pred)
        ground = torch.cat(ground).numpy()
        # print(ground)
        return perf.agg(), pred, ground   
    
         
def check_out_fmt(task, out, dataset_name):
    if task == 'cls':
        assert set(out.keys()) == {'logit'}
        logit = out['logit']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    elif task == 'cls_trans':
        assert set(out.keys()) == {'logit', 'trans_feat'}
        logit = out['logit']
        trans_feat = out['trans_feat']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert isinstance(trans_feat, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert len(trans_feat.shape) == 3
        assert trans_feat.shape[0] == logit.shape[0]
        # 64 coming from pointnet implementation
        assert (trans_feat.shape[1] == trans_feat.shape[2]) and (trans_feat.shape[1] == 64)
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    else:
        assert NotImplemented
        
def get_loss(task, loss_name, data_batch, out, dataset_name):
    """
    Returns the tensor loss function
    :param task:
    :param loss_name:
    :param data_batch: batched data; note not applied data_batch
    :param out: output from the model
    :param dataset_name:
    :return: tensor
    """
    check_out_fmt(task, out, dataset_name)
    if task == 'cls':
        label = data_batch['label'].to(out['logit'].device)
        if loss_name == 'cross_entropy':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'],torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0), label_2[i].unsqueeze(0).long()) * data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'], label_2) * data_batch['lam']
            else:
                loss = F.cross_entropy(out['logit'], label)
        # source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
        elif loss_name == 'smooth':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'],torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0), label_2[i].unsqueeze(0).long()) * data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'], label_2) * data_batch['lam']
            else:
                loss = smooth_loss(out['logit'], label)
        else:
            assert False
    elif task == 'cls_trans':
        label = data_batch['label'].to(out['logit'].device)
        trans_feat = out['trans_feat']
        logit = out['logit']
        if loss_name == 'cross_entropy':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'],torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0), label_2[i].unsqueeze(0).long()) * data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'], label_2) * data_batch['lam']
            else:
                loss = F.cross_entropy(out['logit'], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        elif loss_name == 'smooth':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'],torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0), label_2[i].unsqueeze(0).long()) * data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'], label_2) * data_batch['lam']
            else:
                loss = smooth_loss(out['logit'], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        else:
            assert False
    else:
        assert False

    return loss


def save_checkpoint(id, epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg):
    model.cpu()
    path = f"./runs/{cfg.EXP.EXP_ID}/model_{id}.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_sched_state': lr_sched.state_dict(),
        'bnm_sched_state': bnm_sched.state_dict() if bnm_sched is not None else None,
        'test_perf': test_perf,
    }, path)
    print('Checkpoint saved to %s' % path)
    model.to(DEVICE)


def load_best_checkpoint(model, cfg):
    path = f"./runs/{cfg.EXP.EXP_ID}/model_best.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % path)


def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path):
    print(f'Recovering model and checkpoint from {model_path}')
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state'])
            model = model.module

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    # for backward compatibility with saved models
    if 'lr_sched_state' in checkpoint:
        lr_sched.load_state_dict(checkpoint['lr_sched_state'])
        if checkpoint['bnm_sched_state'] is not None:
            bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
    else:
        print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

    return model


def get_model(cfg):
    print(cfg.EXP.MODEL_NAME)
    if cfg.EXP.MODEL_NAME == 'pointnet2':
        model = models.PointNet2(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.PN2)
    elif cfg.EXP.MODEL_NAME == 'pointnet':
        model = models.PointNet(
            task = cfg.EXP.TASK,
            dataset = cfg.EXP.DATASET)
    elif cfg.EXP.MODEL_NAME == 'dgcnn':
        model = models.DGCNN(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET)
    elif cfg.EXP.MODEL_NAME == 'curvenet':
        model = models.CurveNet(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET)
    elif cfg.EXP.MODEL_NAME == 'simpleview':
        model = models.MVModel(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.MV)
    elif cfg.EXP.MODEL_NAME == 'rscnn':
        model = models.RSCNN(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.RSCNN)
    elif cfg.EXP.MODEL_NAME == 'pct':
        model = models.PCT(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET)
    else:
        assert False

    return model

def get_optimizer(optim_name, tr_arg, model):
    if optim_name == 'vanilla':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=tr_arg.lr_decay_factor,
            patience=tr_arg.lr_reduce_patience,
            verbose=True,
            min_lr=tr_arg.lr_clip)
        bnm_sched = None
    elif optim_name == 'pct':
        pass
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.CosineAnnealingLR(
            optimizer,
            tr_arg.num_epochs,
            eta_min=tr_arg.learning_rate)
        bnm_sched = None
    else:
        assert False

    return optimizer, lr_sched, bnm_sched

def entry_test(cfg, test_or_valid, model_path="", confusion = False,file_object=''):
    split = "test" if test_or_valid else "valid"
    loader_test = create_dataloader(split=split, cfg=cfg)

    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)
    model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path)
    model.eval()
    if confusion:
        test_perf, pred, ground = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT, confusion)
        print(pred.shape, ground.shape)
        #### some hardcoding #######
        np.save('./output/' + cfg.EXP.MODEL_NAME + '_' +  cfg.DATALOADER.MODELNET40_C.corruption + '_' + str(cfg.DATALOADER.MODELNET40_C.severity)  + '_pred.npy',pred )
        np.save('./output/' + cfg.EXP.MODEL_NAME + '_' +  cfg.DATALOADER.MODELNET40_C.corruption + '_' + str(cfg.DATALOADER.MODELNET40_C.severity)  + '_ground.npy',ground)
        #### #### #### #### #### ####
    else:
        test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT, confusion)
    print("Model: {} Corruption: {} Severity: {} Acc: {} Class Acc: {}".format(cfg.EXP.MODEL_NAME, cfg.DATALOADER.MODELNET40_C.corruption, cfg.DATALOADER.MODELNET40_C.severity,test_perf['acc'],test_perf['class_acc']),file=file_object,flush=True)
    pprint.pprint(test_perf, width=80)
    return test_perf

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
        
    file_object = open(args.output, 'a')
    assert not args.exp_config == ""
    assert not args.model_path == ""

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.exp_config)
    if cfg.EXP.DATASET == "modelnet40_c":
        cfg.DATALOADER.MODELNET40_C.corruption = args.corruption
        cfg.DATALOADER.MODELNET40_C.severity = args.severity
    cfg.freeze()
    print(cfg)

    random.seed(cfg.EXP.SEED)
    np.random.seed(cfg.EXP.SEED)
    torch.manual_seed(cfg.EXP.SEED)

    test_or_valid = args.entry == "test"
    entry_test(cfg, test_or_valid, args.model_path, args.confusion,file_object)

if __name__ == '__main__':
    args = parse_args()
    main(args)