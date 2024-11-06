import torch
import torch.optim as optim
import torch.nn.functional as F  # Add this line
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from time import time
import pprint
import matplotlib.pyplot as plt
import os
import open3d as o3d
from dataset import Dataset
from pointnet_pyt.pointnet.model import PointNetDenseCls, feature_transform_regularizer  # Assuming pointnet.py contains your models
from pointnet2_pyt.pointnet2.models import PointNet2PartSeg
from PCT.model import PCT_SEG, PCT_NEW
from progressbar import ProgressBar
from dgcnn.model import DGCNNSegmentation

def save_predictions_as_off(points, labels, preds, file_path):
    # Open the file in write mode
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(points)} 0 0\n")  # Number of vertices, no faces, no edges
    
        # Write each point with its color based on the predicted label
        for i in range(len(points)):
            x, y, z = points[i][:3]
            
            gt_label = labels[i]
            
            segment_label = preds[i]
            f.write(f"{x} {y} {z} {gt_label} {segment_label}\n")

    print(f"Saved predictions to {file_path}")
    
# Step 1: Hyperparameters
batch_size = 16
num_point = 2048
learning_rate = 0.001
num_epochs = 1
num_classes = 50  # For ShapeNetPart

# Step 2: Define the dataset paths
root_dir = '/home/yangqing/Documents/My_PointCloud_Model/data/'

# Step 3: Set up data loaders
train_dataset = Dataset(root=root_dir, dataset_name='shapenetpart', num_points=num_point, split='train', segmentation=True)
test_dataset = Dataset(root=root_dir, dataset_name='shapenetpart', num_points=num_point, split='test', segmentation=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Initialize the model, loss function, and optimizer# Step 4: Define model selection
model_name = 'pct'  # Options: 'pointnet', 'pointnet2', 'dgcnn', 'pct'

if model_name == 'pointnet':
    model = PointNetDenseCls(k=num_classes, feature_transform=True)
elif model_name == 'pointnet2':
    model = PointNet2PartSeg(num_class=num_classes, normal_channel=False)
elif model_name == 'dgcnn':
    model = DGCNNSegmentation(num_classes=num_classes)
elif model_name == 'pct':
    model = PCT_NEW()

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
        
        if model_name == 'pointnet' or model_name == 'pct':
            point_set = point_set.permute(0, 2, 1)
        
        point_set = point_set.cuda()
        
        seg = seg.cuda()

        optimizer.zero_grad()
        if model_name == 'pct':
            pred, trans, trans_feat = model(point_set)
        else:
            pred, trans, trans_feat = model(point_set)

        # Compute loss without reshaping pred
        if model_name == 'dgcnn' or model_name == 'pct':
            loss = F.cross_entropy(pred.reshape(-1, num_classes), seg.reshape(-1))
        else: 
            loss = F.cross_entropy(pred.view(-1, num_classes), seg.view(-1))  # Flatten for per-point cross-entropy
        
        if trans_feat != None and model.feature_transform:
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
    
class PerfTrackVal:
    def __init__(self, task, extra_param=None):
        self.total_loss = 0
        self.total_correct = 0
        self.total_seen = 0
        self.task = task
        self.extra_param = extra_param
        self.intersections = np.zeros(num_classes)  # To store intersections per class
        self.unions = np.zeros(num_classes)  # To store unions per class
        self.num_classes = num_classes

    def update(self, data_batch, out):
        # Assuming `out` contains 'pred' and 'seg' for predictions and ground truth
        pred = out['pred']
        seg = out['seg']

        # Compute loss (assuming it's cross-entropy loss for this example)
        loss = F.cross_entropy(pred.permute(0, 2, 1), seg)
        self.total_loss += loss.item()

        # Compute accuracy
        pred_choice = pred.max(2)[1]
        correct = pred_choice.eq(seg).sum().item()
        self.total_correct += correct
        self.total_seen += seg.numel()
        
        # Calculate IoU for each class
        for class_id in range(self.num_classes):
            # Boolean tensors for intersections and unions
            pred_mask = (pred_choice == class_id)
            seg_mask = (seg == class_id)
            intersection = (pred_mask & seg_mask).sum().item()
            union = (pred_mask | seg_mask).sum().item()
            
            # Accumulate intersections and unions per class
            self.intersections[class_id] += intersection
            self.unions[class_id] += union

    def agg(self):
        # Aggregate metrics (accuracy and average loss)
        avg_loss = self.total_loss / self.total_seen if self.total_seen > 0 else 0
        accuracy = self.total_correct / self.total_seen if self.total_seen > 0 else 0
        iou_per_class = self.intersections / (self.unions + 1e-6)  # Avoid division by zero
        mean_iou = np.nanmean(iou_per_class)  # Compute the mean ignoring NaNs
        return {'mIoU': mean_iou, 'avg_loss': avg_loss, 'accuracy': accuracy}

def test_model(task, loader, model, dataset_name, num_classes, confusion=False, save_dir="./predictions"):
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
            
            print("185: ",point_set.shape)
            point_set = point_set.cuda()
            seg = seg.cuda()

            
            if model_name == 'pointnet' or model_name == 'pct':
                point_set = point_set.permute(0, 2, 1)  # Transpose to [batch_size, 3, num_points]

            time2 = time()

            # Forward pass
            pred, trans, trans_feat = model(point_set)  # Shape: [batch_size, num_points, num_classes]

            print("198 pred:",pred.shape)
            # Get the per-point predicted labels
            pred_choice = pred.max(2)[1].cpu().numpy()  # Shape: [batch_size, num_points]
            print("200 pred:",pred_choice.shape)
            time3 = time()
            perf.update(data_batch=data_batch, out={'pred': pred, 'seg': seg})
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()
            bar.update(i)
            
            print(point_set.shape)
            print(pred_choice.shape)
            print(seg.shape)
            
            # Save each batch of predictions to a file
            for j in range(point_set.shape[0]):
                file_path = os.path.join(save_dir, model_name, f"pred_batch{i}_point{j}.off")
                
                if model_name == 'pointnet' or model_name == 'pct':
                    points = point_set[j].cpu().numpy().T
                else:
                    points = point_set[j].cpu().numpy()
                save_predictions_as_off(
                    points=points, # Point cloud coordinates
                    labels=seg[j].cpu().numpy(),        # Ground truth labels
                    preds=pred_choice[j],               # Predicted labels (already NumPy array)
                    file_path=file_path
                )

    print(f"Time DL: {time_dl:.4f}, Time Get Inp: {time_gi:.4f}, Time Model: {time_model:.4f}, Time Update: {time_upd:.4f}")
    
    return perf.agg()



# Call test_model() after training to evaluate
perf = test_model(task='part_segmentation', loader=test_loader, model=model, dataset_name='shapenetpart', num_classes=50, confusion=True)
pprint.pprint(perf, width=80)