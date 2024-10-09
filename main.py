# from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import logging
from tqdm import tqdm
from dataloader import create_dataloader
import sys
from collections import defaultdict
import importlib
from all_utils import (
    TensorboardManager, PerfTrackTrain,
    PerfTrackVal, TrackTrain, smooth_loss, DATASET_NUM_CLASS,
    rscnn_voting_evaluate_cls, pn2_vote_evaluate_cls)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--entry', type=str, default="train")
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default="log", help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()

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


def entry_train(cfg, resume=False, model_path=""):
    loader_train = create_dataloader(split='train', cfg=cfg)
    print(loader_train)
    return
    loader_valid = create_dataloader(split='valid', cfg=cfg)
    loader_test  = create_dataloader(split='test',  cfg=cfg)

    model = get_model(cfg)
    return 
    model.to(DEVICE)
    print(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)

    if resume:
        model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path)
    else:
        assert model_path == ""

    log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb = TensorboardManager(log_dir)
    track_train = TrackTrain(early_stop_patience=cfg.TRAIN.early_stop)

    for epoch in range(cfg.TRAIN.num_epochs):
        print(f'Epoch {epoch}')

        print('Training..')
        train_perf, train_loss = train(cfg.EXP.TASK, loader_train, model, optimizer, cfg.EXP.LOSS_NAME, cfg.EXP.DATASET, cfg)
        pprint.pprint(train_perf, width=80)
        tb.update('train', epoch, train_perf)

        if (not cfg.EXP_EXTRA.no_val) and epoch % cfg.EXP_EXTRA.val_eval_freq == 0:
                print('\nValidating..')
                val_perf = validate(cfg.EXP.TASK, loader_valid, model, cfg.EXP.DATASET, cfg.ADAPT)
                pprint.pprint(val_perf, width=80)
                tb.update('val', epoch, val_perf)
        else:
            val_perf = defaultdict(float)

        if (not cfg.EXP_EXTRA.no_test) and (epoch % cfg.EXP_EXTRA.test_eval_freq == 0):
            print('\nTesting..')
            test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT)
            pprint.pprint(test_perf, width=80)
            tb.update('test', epoch, test_perf)
        else:
            test_perf = defaultdict(float)

        track_train.record_epoch(
            epoch_id=epoch,
            train_metric=get_metric_from_perf(cfg.EXP.TASK, train_perf, cfg.EXP.METRIC),
            val_metric=get_metric_from_perf(cfg.EXP.TASK, val_perf, cfg.EXP.METRIC),
            test_metric=get_metric_from_perf(cfg.EXP.TASK, test_perf, cfg.EXP.METRIC))

        if (not cfg.EXP_EXTRA.no_val) and track_train.save_model(epoch_id=epoch, split='val'):
            print('Saving best model on the validation set')
            save_checkpoint('best_val', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_test) and track_train.save_model(epoch_id=epoch, split='test'):
            print('Saving best model on the test set')
            save_checkpoint('best_test', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_val) and track_train.early_stop(epoch_id=epoch):
            print(f"Early stopping at {epoch} as val acc did not improve for {cfg.TRAIN.early_stop} epochs.")
            break

        if (not (cfg.EXP_EXTRA.save_ckp == 0)) and (epoch % cfg.EXP_EXTRA.save_ckp == 0):
            save_checkpoint(f'{epoch}', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if cfg.EXP.OPTIMIZER == 'vanilla':
            assert bnm_sched is None
            lr_sched.step(train_loss)
        else:
            lr_sched.step()
            
    print('Saving the final model')
    save_checkpoint('final', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

    print('\nTesting on the final model..')
    last_test_perf = validate(cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT)
    pprint.pprint(last_test_perf, width=80)

    tb.close()

def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


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
        
    ## test_dataset = ModelNetDataLoader(root=data_path, split='test', process_data=False)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    # '''MODEL LOADING'''
    # num_class = args.num_category
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    # model = importlib.import_module(model_name)

    # classifier = model.get_model(num_class, normal_channel=args.use_normals)
    # if not args.use_cpu:
    #     classifier = classifier.cuda()

    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])

    # with torch.no_grad():
    #     instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
    #     log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)