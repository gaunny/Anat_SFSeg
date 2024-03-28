import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from torch import sigmoid

class PointNetfeat(nn.Module):
    def __init__(self,use_conv=True):
        super(PointNetfeat, self).__init__()
        # 3-layer MLP (via 1D-CNN) : encoder points individually
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128,1024, 1)
        if use_conv:
            self.conv4 = torch.nn.Conv1d(3,128,1)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.conv4:
            identity = self.conv4(identity)
        x += identity
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class SegNet(nn.Module):
    def __init__(self, seg_cls,k=198):  
        super(SegNet,self).__init__()
        if seg_cls == 'swm':
            self.emb_layer1 = nn.Embedding(8,120)  #emb_dim=8 for swm 
        if seg_cls == 'dwm':
            self.emb_layer1 = nn.Embedding(34,120)   #embdim=34 for dwm
        self.emb_layer2 = nn.Embedding(105,7)
        self.feat1 = PointNetfeat()
        self.fc1 = nn.Linear(1024+120+7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, point,group_FiberAnatMap,ind_FiberAnatMap):  
        group_FiberAnatMap = group_FiberAnatMap.int()
        group_FiberAnatMap_ = group_FiberAnatMap.view(group_FiberAnatMap.size()[0],1) 
        ind_FiberAnatMap_ = torch.squeeze(ind_FiberAnatMap)
        if ind_FiberAnatMap_.dim() == 1:
            ind_FiberAnatMap_ = ind_FiberAnatMap_.unsqueeze(0)

        emb1 = self.emb_layer1(group_FiberAnatMap_)
        emb1 = emb1.mean(dim=1)
        input_tensor = ind_FiberAnatMap_.long()
        emb2 = self.emb_layer2(input_tensor)
        emb2 = emb2.mean(dim=1) 

        out = torch.cat([self.feat1(point),emb1],dim=1)
        out = torch.cat([out,emb2],dim=1)
        out = F.relu(self.bn1(self.fc1(out)))
        out= F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.dropout(self.fc3(out))))
        out = self.fc4(out)
       
        return F.log_softmax(out, dim=1)  
