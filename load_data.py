import h5py
import os
import numpy as np
import torch
import torch.utils.data as data

class WMASWMDataset(data.Dataset):
    def __init__(self, root, logger, num_fold=1, k=5, split='train'):
        self.split = split
        self.num_fold = num_fold
        self.k = k
        self.logger = logger
        points_combine = None
        labels_combine = None
        group_FiberAnatMaps_combine = None
        ind_FiberAnatMaps_combine = None
        if self.split == 'train':
            train_fold = 0
            train_fold_lst = []
            for i in range(5):
                if i+1 != self.num_fold:
                    feat_h5 = h5py.File(os.path.join(root ,'sf_clusters_train_featMatrix_{}.h5'.format(str(i+1))), 'r')
                    points = np.array(feat_h5['point'])
                    labels = np.array(feat_h5['label'])
                    group_FiberAnatMaps = np.array(feat_h5['region'])
                    ind_FiberAnatMaps = np.array(feat_h5['ind_region'])
                    if train_fold == 0:
                        points_combine = points
                        labels_combine = labels
                        group_FiberAnatMaps_combine = group_FiberAnatMaps
                        ind_FiberAnatMaps_combine = ind_FiberAnatMaps
                    else:
                        points_combine = np.concatenate((points_combine, points), axis=0)
                        labels_combine = np.concatenate((labels_combine, labels), axis=0)
                        group_FiberAnatMaps_combine = np.concatenate((group_FiberAnatMaps_combine, group_FiberAnatMaps), axis=0)
                        ind_FiberAnatMaps_combine = np.concatenate((ind_FiberAnatMaps_combine,ind_FiberAnatMaps),axis=0)
                    train_fold_lst.append(i+1)
                    train_fold += 1
            self.points = points_combine
            self.labels = labels_combine
            self.group_FiberAnatMaps = group_FiberAnatMaps_combine
            self.ind_FiberAnatMaps = ind_FiberAnatMaps_combine
            logger.info('use {} fold as train data'.format(train_fold_lst))
            logger.info('The size of feature for {} is {}'.format(self.split, self.points.shape))
        else:
            feat_h5 = h5py.File(os.path.join(root,'sf_clusters_train_featMatrix_{}.h5'.format(self.num_fold)), 'r')
            self.points = np.array(feat_h5['point'])
            self.labels = np.array(feat_h5['label'])
            self.group_FiberAnatMaps = np.array(feat_h5['region'])
            self.ind_FiberAnatMaps = np.array(feat_h5['ind_region'])
           
            logger.info('use {} fold as validation data'.format(self.num_fold))
            logger.info('The size of feature for {} is {}'.format(self.split, self.points.shape))

        # label names list
        self.label_names = [*feat_h5['label_name']]

    def __getitem__(self, index):
        point_set = self.points[index]
        label = self.labels[index]
        group_FiberAnatMap = self.group_FiberAnatMaps[index]
        ind_FiberAnatMap = self.ind_FiberAnatMaps[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            print('Point is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            print('Label is not in int64 format')
        if group_FiberAnatMap.dtype == 'int64':
            group_FiberAnatMap = torch.from_numpy(np.array([group_FiberAnatMap]))
        else:
            group_FiberAnatMap = torch.from_numpy(np.array([group_FiberAnatMap]).astype(np.int64))
        if ind_FiberAnatMap.dtype == 'int64':
            ind_FiberAnatMap = torch.from_numpy(np.array([ind_FiberAnatMap]))
        else:
            ind_FiberAnatMap = torch.from_numpy(np.array([ind_FiberAnatMap]).astype(np.int64))
            # print('ind_FiberAnatMap is not in int64 format')
        return point_set, label,group_FiberAnatMap,ind_FiberAnatMap

    def __len__(self):
        return len(self.labels)

    def obtain_label_names(self):
        return self.label_names
    
class WMADWMDataset(data.Dataset):
    def __init__(self, root, logger, num_fold=1, k=5, split='train'):
        self.split = split
        self.num_fold = num_fold
        self.k = k
        self.logger = logger
        points_combine = None
        labels_combine = None
        group_FiberAnatMaps_combine = None
        ind_FiberAnatMaps_combine = None
        if self.split == 'train':
            train_fold = 0
            train_fold_lst = []
            for i in range(5):
                if i+1 != self.num_fold:
                    feat_h5 = h5py.File(os.path.join(root ,'sf_clusters_train_featMatrix_{}.h5'.format(str(i+1))), 'r')
                    points = np.array(feat_h5['point'])
                    labels = np.array(feat_h5['label'])
                    group_FiberAnatMaps = np.array(feat_h5['region'])
                    ind_FiberAnatMaps = np.array(feat_h5['ind_region'])
                    # the labels in WMA-DWM dataset should minus 198 to accurately load
                    labels = labels-198
                    
                    if train_fold == 0:
                        points_combine = points
                        labels_combine = labels
                        group_FiberAnatMaps_combine = group_FiberAnatMaps
                        ind_FiberAnatMaps_combine = ind_FiberAnatMaps
                    else:
                        points_combine = np.concatenate((points_combine, points), axis=0)
                        labels_combine = np.concatenate((labels_combine, labels), axis=0)
                        group_FiberAnatMaps_combine = np.concatenate((group_FiberAnatMaps_combine, group_FiberAnatMaps), axis=0)
                        ind_FiberAnatMaps_combine = np.concatenate((ind_FiberAnatMaps_combine,ind_FiberAnatMaps),axis=0)
                    train_fold_lst.append(i+1)
                    train_fold += 1
            self.points = points_combine
            self.labels = labels_combine
            self.group_FiberAnatMaps = group_FiberAnatMaps_combine
            self.ind_FiberAnatMaps = ind_FiberAnatMaps_combine
            logger.info('use {} fold as train data'.format(train_fold_lst))
            logger.info('The size of feature for {} is {}'.format(self.split, self.points.shape))
        else:
            feat_h5 = h5py.File(os.path.join(root,'sf_clusters_train_featMatrix_{}.h5'.format(self.num_fold)), 'r')
            self.points = np.array(feat_h5['point'])
            self.labels = np.array(feat_h5['label'])
            self.group_FiberAnatMaps = np.array(feat_h5['region'])
            self.ind_FiberAnatMaps = np.array(feat_h5['ind_region'])
            self.labels = self.labels -198
           
            logger.info('use {} fold as validation data'.format(self.num_fold))
            logger.info('The size of feature for {} is {}'.format(self.split, self.points.shape))

        # label names list
        self.label_names = [*feat_h5['label_name']]
        
    def __getitem__(self, index):
        point_set = self.points[index]
        label = self.labels[index]
        group_FiberAnatMap = self.group_FiberAnatMaps[index]
        ind_FiberAnatMap = self.ind_FiberAnatMaps[index]
        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            print('Point is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            print('Label is not in int64 format')
        if group_FiberAnatMap.dtype == 'int64':
            group_FiberAnatMap = torch.from_numpy(np.array([group_FiberAnatMap]))
        else:
            group_FiberAnatMap = torch.from_numpy(np.array([group_FiberAnatMap]).astype(np.int64))
        if ind_FiberAnatMap.dtype == 'int64':
            ind_FiberAnatMap = torch.from_numpy(np.array([ind_FiberAnatMap]))
        else:
            ind_FiberAnatMap = torch.from_numpy(np.array([ind_FiberAnatMap]).astype(np.int64))
            # print('ind_FiberAnatMap is not in int64 format')
        return point_set, label,group_FiberAnatMap,ind_FiberAnatMap#,FiberAnatMap,ind_FiberAnatMap

    def __len__(self):
        return len(self.labels)

    def obtain_label_names(self):
        return self.label_names