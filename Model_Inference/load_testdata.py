import numpy as np
import torch
import re
import glob
import torch.utils.data as data
import h5py
import os
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

class SWMDWM_ClsDataset(data.Dataset):
    def __init__(self,num,input_path):  #num
        paths = natural_sort(glob.glob(f'{input_path}/*/'))
        points_combine = []
        for i in range(len(paths)):
            points_file = os.path.join(paths[i],'downsampled.npy')
            points = np.load(points_file)
            points_combine.append(points)
        self.points = points_combine[num]
    def __getitem__(self,index):
        point_set = self.points[index]  
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32)) 
        return point_set
    def __len__(self):
        return self.points.shape[0]
    
class SWMOutliers_ClsDataset(data.Dataset):
    def __init__(self,num,input_path): 
        paths = natural_sort(glob.glob(f'{input_path}/*/'))
        features_combine = []
        for i in range(len(paths)):
            points_file = os.path.join(paths[i],'point_swm.npy')
            points = np.load(points_file)
            features_combine.append(points)
        self.points = features_combine[num]

    def __getitem__(self,index):
        point_set = self.points[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))    
        return point_set
    def __len__(self):
        return self.points.shape[0]

class SegDataset(data.Dataset):
    def __init__(self,num,input_path,seg_cls):
        paths = natural_sort(glob.glob(f'{input_path}/*/')) 
        points_combine = []
        ind_FiberAnatMap_combine = []
        group_FiberAnatMap_combine = []
        for i in range(len(paths)):
            if seg_cls=='swm':
                h5_file = os.path.join(paths[i],'move_outlier_swm.h5')
            if seg_cls=='dwm':
                h5_file = os.path.join(paths[i],'dwm_featMatrix.h5')
            feat_h5 = h5py.File(h5_file,'r')
            points_combine.append(np.array(feat_h5['point']))
            ind_FiberAnatMap_combine.append(np.array(feat_h5['ind_FiberAnatMap']))
            group_FiberAnatMap_combine.append(np.array(feat_h5['group_FiberAnatMap']))
        self.points = points_combine[num]
        self.ind_FiberAnatMap = ind_FiberAnatMap_combine[num]
        self.group_FiberAnatMap = group_FiberAnatMap_combine[num]
        print('self.ind_FiberAnatMap',self.ind_FiberAnatMap.shape)

    def __getitem__(self,index):
        point = self.points[index]
        ind_FiberAnatMap = self.ind_FiberAnatMap[index]
        group_FiberAnatMap = self.group_FiberAnatMap[index]
        if point.dtype == 'float32':
            point = torch.from_numpy(point)
        else:
            point = torch.from_numpy(point.astype(np.float32))
        if ind_FiberAnatMap.dtype == 'int64':
            ind_FiberAnatMap = torch.from_numpy(np.array([ind_FiberAnatMap]))
        else:
            ind_FiberAnatMap = torch.from_numpy(np.array([ind_FiberAnatMap]).astype(np.int64))
        if group_FiberAnatMap.dtype == 'int64':
            group_FiberAnatMap = torch.from_numpy(np.array([group_FiberAnatMap]))
        else:
            group_FiberAnatMap = torch.from_numpy(np.array([group_FiberAnatMap]).astype(np.int64))
        return point,group_FiberAnatMap,ind_FiberAnatMap  
    def __len__(self):
        return self.points.shape[0]
