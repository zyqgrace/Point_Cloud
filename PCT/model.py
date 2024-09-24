import torch
import torch.nn as nn
import torch.nn.functional as F
from PCT.utils import sample_and_group 


class Encoder_Layer(nn.Module):
    def __init__(self, args, channels=256):
        super(Encoder_Layer,self).__init__()
        self.args = args 
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.SA1 = Self_Attention_Layer(channels)
        self.SA2 = Self_Attention_Layer(channels)
        self.SA3 = Self_Attention_Layer(channels)
        self.SA4 = Self_Attention_Layer(channels)

    def forward(self,x):
        # b, 3, npoint, nsample 
        # conv2d 3 -> 128 channels, 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.SA1(x)
        x2 = self.SA2(x1)
        x3 = self.SA2(x2)
        x4 = self.SA2(x3)
        x = torch.cat((x1, x2, x3, x4), dim = 1)
        return x
        
class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x
    
    
class PCT(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PCT, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.encode = Encoder_Layer(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        
    def forward(self, x):
        xyz = x.permute(0,2,1)
        
        batch_size, _, _ = x.size()
        
        #B, D. M
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        xyz = xyz.contiguous()
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        
        x = self.encode(feature_1)
        x = torch.cat([x, feature_1],dim = 1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
        
        
class Self_Attention_Layer(nn.Module):
    def __init__(self, channels):
        super(Self_Attention_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels //4, 1, bias = False)
        self.k_conv = nn.Conv1d(channels, channels //4, 1, bias = False)
        self.q_conv.weight = self.q_conv.weight
        self.k_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        # b, n, c --> [N, 128]
        Q = self.q_conv(x).permute(0,2,1)
        
        # b, c, n --> [128, N]
        K = self.k_conv(x)
        
        # b, c, n
        V = self.v_conv(x)
        
        # matmul --> b, n, n
        similarity = torch.bmm(Q, K)
        
        attention = self.softmax(similarity)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        
        # b, c, n 
        R = torch.bmm(V, attention)
        
        # offset atention 
        R = self.act(self.after_norm(self.trans_conv(x - R)))
        
        x = x + R
        return x 