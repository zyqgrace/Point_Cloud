import torch
import torch.optim as optim
import torch.nn.functional as F  # Add this line
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from time import time
import pprint

from dataset import Dataset
from pointnet_pyt.pointnet.model import PointNetDenseCls, feature_transform_regularizer  # Assuming pointnet.py contains your models
from pointnet2_pyt.pointnet2.models import PointNet2PartSeg
from progressbar import ProgressBar

# Step 1: Hyperparameters
batch_size = 32
num_point = 2048
learning_rate = 0.001
num_epochs = 10
num_classes = 50  # For ShapeNetPart

# Step 2: Define the dataset paths
root_dir = '/home/yangqing/Documents/My_PointCloud_Model/data/'

# Step 3: Set up data loaders
train_dataset = Dataset(root=root_dir, dataset_name='shapenetpart', num_points=2048, split='train', segmentation=True)
test_dataset = Dataset(root=root_dir, dataset_name='shapenetpart', num_points=2048, split='test', segmentation=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Initialize the model, loss function, and optimizer# Step 4: Define model selection
model_name = 'pointnet'  # Options: 'pointnet', 'pointnet2'

if model_name == 'pointnet':
    model = PointNetDenseCls(k=num_classes, feature_transform=True)
elif model_name == 'pointnet2':
    model = PointNet2PartSeg(num_class=num_classes, normal_channel=False)


model = model.cuda()

# Step 5: Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Step 3: Define the training loop
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_seen = 0
    
    for i, (point_set, label, seg, name, file) in enumerate(train_loader):
        
        if model_name == 'pointnet':
            point_set = point_set.permute(0, 2, 1)
        
        point_set = point_set.cuda()
        
        seg = seg.cuda()

        optimizer.zero_grad()
        pred, trans, trans_feat = model(point_set)

        # Compute loss without reshaping pred
        loss = F.nll_loss(pred.permute(0, 2, 1), seg) 
        if model.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred_choice = pred.max(2)[1]  # Get the class with max probability for each point
        correct = pred_choice.eq(seg).sum().item()
        total_correct += correct
        total_seen += seg.numel()
        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

# Step 4: Train the model
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    
# Step 5: Evaluate the model on the test set (Optional)
def test_model():
    model.eval()
    total_correct = 0
    total_seen=0

    with torch.no_grad():  # Disable gradient computation for inference
        for i, (point_set, label, seg, name, file) in enumerate(test_loader):
            if model_name == 'pointnet':
                point_set = point_set.permute(0, 2, 1)  # Transpose to [batch_size, 3, num_points]       
                point_set = point_set.cuda()
                seg = seg.cuda()

                # Forward pass
                pred, _, _ = model(point_set)

                # Flatten predictions and segmentation labels
                pred = pred.view(-1, num_classes)
                seg = seg.view(-1)

                # Get predicted labels
                pred_choice = pred.max(1)[1]  # Get the class with the max probability
                correct = pred_choice.eq(seg).sum().item()
                total_correct += correct
                total_seen += seg.numel()
            else:
                point_set = point_set.cuda()
                seg = seg.cuda()

                # Forward pass
                pred, _, _ = model(point_set)

                pred_choice = pred.data.max(2)[1]  # Get predicted labels
                correct = pred_choice.eq(seg.data).cpu().sum()
                total_correct += correct.item()
                total_seen += seg.numel()
    print(f"Test Accuracy: {total_correct / total_seen:.4f}")

class PerfTrackVal:
    def __init__(self, task, extra_param=None):
        self.total_loss = 0
        self.total_correct = 0
        self.total_seen = 0
        self.task = task
        self.extra_param = extra_param

    def update(self, data_batch, out):
        # Assuming `out` contains 'pred' and 'seg' for predictions and ground truth
        pred = out['pred']
        seg = out['seg']

        # Compute loss (assuming it's cross-entropy loss for this example)
        loss = nn.functional.cross_entropy(pred.permute(0, 2, 1), seg)
        self.total_loss += loss.item()

        # Compute accuracy
        pred_choice = pred.max(2)[1]
        correct = pred_choice.eq(seg).sum().item()
        self.total_correct += correct
        self.total_seen += seg.numel()

    def agg(self):
        # Aggregate metrics (accuracy and average loss)
        avg_loss = self.total_loss / self.total_seen if self.total_seen > 0 else 0
        accuracy = self.total_correct / self.total_seen if self.total_seen > 0 else 0
        return {'avg_loss': avg_loss, 'accuracy': accuracy}

def test_model2(task, loader, model, dataset_name, num_classes, confusion=False):
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
        time5 = time()
        if confusion:
            pred = []
            ground = []
        for i, data_batch in enumerate(loader):
            time1 = time()
            
            # Unpack the data_batch tuple
            point_set, label, seg, name, file = data_batch
            
            point_set = point_set.cuda()
            seg = seg.cuda()
            
            if model_name == 'pointnet':
                point_set = point_set.permute(0, 2, 1)  # Transpose to [batch_size, 3, num_points]

            time2 = time()

            # Forward pass
            out = model(point_set)

            out = model(point_set)
            pred_tensor = out[0]
            pred_tensor = pred_tensor.view(-1, num_classes)
            seg = seg.view(-1)

            if confusion:
                # Convert to CPU and append to lists directly
                pred_choice = pred_tensor.max(1)[1].cpu().numpy()
                seg_np = seg.cpu().numpy()

                pred.append(pred_choice)
                ground.append(seg_np)

            time3 = time()
            perf.update(data_batch=data_batch, out={'pred': pred_tensor, 'seg': seg})
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()
            bar.update(i)

    print(f"Time DL: {time_dl:.4f}, Time Get Inp: {time_gi:.4f}, Time Model: {time_model:.4f}, Time Update: {time_upd:.4f}")
    
    if not confusion:
        return perf.agg()
    else:
        # Convert lists to numpy arrays and concatenate them
        pred = np.concatenate(pred, axis=0)
        ground = np.concatenate(ground, axis=0)
        return perf.agg(), pred, ground



# Call test_model() after training to evaluate
perf = test_model2(task='part_segmentation', loader=test_loader, model=model, dataset_name='shapenetpart', num_classes=50, confusion=True)
pprint.pprint(perf, width=80)