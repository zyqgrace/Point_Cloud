import torch
import torch.optim as optim
import torch.nn.functional as F  # Add this line
import tensorflow as tf
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Import the dataset class and models
from dataset import Dataset
from pointnet_pyt.pointnet.model import PointNetDenseCls, feature_transform_regularizer  # Assuming pointnet.py contains your models

# Step 1: Hyperparameters
batch_size = 32
num_point = 2048
learning_rate = 0.001
num_epochs = 30
num_classes = 50  # For ShapeNetPart

# Step 2: Define the dataset paths
root_dir = '/home/yangqing/Documents/My_PointCloud_Model/data/'

# Step 3: Set up data loaders
train_dataset = Dataset(root=root_dir, dataset_name='shapenetpart', num_points=2048, split='train', segmentation=True)
test_dataset = Dataset(root=root_dir, dataset_name='shapenetpart', num_points=2048, split='test', segmentation=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Initialize the model, loss function, and optimizer
model = PointNetDenseCls(k=50, feature_transform=True)  # k is the number of classes (50 for ShapeNetPart)
model = model.cuda()

# Step 5: Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Step 3: Define the training loop
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_seen = 0
    
    for i, (point_set, label, seg, name, file) in enumerate(train_loader):
        
        point_set = point_set.permute(0, 2, 1)
        
        point_set = point_set.cuda()
        
        seg = seg.cuda()

        optimizer.zero_grad()

        pred, trans, trans_feat = model(point_set)
        pred = pred.view(-1, 50)  # Flatten for loss computation
        seg = seg.view(-1)  # Flatten segmentation labels to match predictions

        # Compute loss
        loss = F.nll_loss(pred, seg)
        
        # Add regularizer for the transformation matrix
        if model.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred_choice = pred.data.max(1)[1]  # Choose the class with max log probability
        correct = pred_choice.eq(seg.data).cpu().sum()
        total_correct += correct.item()
        total_seen += seg.size(0)
        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

# Step 4: Train the model
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    
# Step 5: Evaluate the model on the test set (Optional)
def test_model():
    model.eval()
    total_correct = 0
    total_points = 0
    total_seen=0

    with torch.no_grad():  # Disable gradient computation for inference
        for i, (point_set, label, seg, name, file) in enumerate(test_loader):
            point_set = point_set.permute(0, 2, 1)  # Transpose to [batch_size, 3, num_points]
            point_set = point_set.cuda()
            seg = seg.cuda()

            # Forward pass
            pred, _, _ = model(point_set)

            pred_choice = pred.data.max(2)[1]  # Get predicted labels
            correct = pred_choice.eq(seg.data).cpu().sum()
            total_correct += correct.item()
            total_seen += point_set.size(0) * point_set.size(2)

    print(f"Test Accuracy: {total_correct / total_seen:.4f}")
    
# Call test_model() after training to evaluate
test_model()