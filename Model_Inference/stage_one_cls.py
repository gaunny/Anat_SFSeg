### Classify the SWM and DWM
import torch
import argparse
import os
import time
import h5py
import re
import glob
import numpy as np
from load_testdata import SWMDWM_ClsDataset
from stage_one_model import PointNetCls

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_model(k,weight_path,best_metric):
    classifer = PointNetCls(k=k).to(device)
    classifer_weight_path = os.path.join(weight_path,'best_{}_model.pth'.format(best_metric))
    classifer.load_state_dict(torch.load(classifer_weight_path))
    return classifer

def test_net(k,weight_path,best_metric,test_data_loader,output_prediction_mask_path):
    print('')
    print('===================================')
    print('')
    classifer_net = load_model(k,weight_path,best_metric)
    start_time = time.time()
    with torch.no_grad():
        test_predicted_lst = []
        for j,data in (enumerate(test_data_loader, 0)):
            points = data
            points = points.transpose(2, 1)  
            points = points.to(device)
            classifer_net = classifer_net.eval()
            pred = classifer_net(points)
            _,pred_idx = torch.max(pred, dim=1)
            pred_idx = torch.where(pred_idx < k, pred_idx, torch.tensor(k).to(device))
            pred_idx = pred_idx.cpu().detach().numpy().tolist()
            test_predicted_lst.extend(pred_idx)
    end_time = time.time()
    print('The total time of prediction is:{} s'.format(round((end_time - start_time), 4)))
    print('The test sample size is: ', len(test_predicted_lst))
    test_prediction_lst_h5 = h5py.File(output_prediction_mask_path,'w')
    test_prediction_lst_h5['complete_pred_test'] = test_predicted_lst
    test_predicted_array = np.asarray(test_predicted_lst)
    return test_predicted_array

def save_results(input_path,output_prediction_mask_path_all,output_path,atlas_cluster):
    paths = natural_sort(glob.glob(f'{input_path}/*/'))
    ind_FiberAnatMap_all = np.load(f'{input_path}/ind_FiberAnatMap_all_subjects.npy')
    print('ind_FiberAnatMap_all',ind_FiberAnatMap_all.shape)
    for m in range(len(paths)):
        subject_name = paths[m].split('/')[-2]
        points_file = os.path.join(paths[m],'downsampled.npy')
        h5_path = output_prediction_mask_path_all[m]
        h5_feat = h5py.File(h5_path,'r')
        test_predicted_lst = np.asarray(h5_feat['complete_pred_test'])
        points = np.load(points_file)
        ind_FiberAnatMaps = ind_FiberAnatMap_all[m,:,0,:]
        point_swm = None 
        point_dwm = None
        ind_FiberAnatMap_swm = None
        ind_FiberAnatMap_dwm = None
        i_swm_lst = []
        # concatenate all 198 SWM clusters
        for i in range(198):
            index = np.where(test_predicted_lst==i)[0]
            point_i = points[index]
            point_i = np.around(point_i,decimals=4)
            ind_FiberAnatMap = ind_FiberAnatMaps[index]
            if point_i.shape[0]!=0:
                i_swm = i
                i_swm_lst.append(i_swm)
                if i ==i_swm_lst[0]:
                    point_swm = point_i
                    ind_FiberAnatMap_swm = ind_FiberAnatMap
                else:
                    point_swm = np.concatenate((point_swm,point_i))
                    ind_FiberAnatMap_swm = np.concatenate((ind_FiberAnatMap_swm,ind_FiberAnatMap))
        # Interpolate 
        repeats = ind_FiberAnatMap_swm.shape[0] / 2000 +1  
        new_arr = np.repeat(atlas_cluster, repeats)
        if ind_FiberAnatMap_swm.shape[0] >= 2000:
            new_arr_ = np.random.choice(new_arr,ind_FiberAnatMap_swm.shape[0] , replace=False)
        if ind_FiberAnatMap_swm.shape[0] < 2000:
            new_arr_ = np.random.choice(new_arr,ind_FiberAnatMap_swm.shape[0] , replace=True)
        
        arrIndex = np.array(new_arr_).argsort()
        group_FiberAnatMap_swm = new_arr_[arrIndex]  
        print('ind_FiberAnatMap_swm',ind_FiberAnatMap_swm.shape)
        print('group_FiberAnatMap_swm',group_FiberAnatMap_swm.shape)
        out_path = os.path.join(output_path,subject_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.save(f'{out_path}/point_swm.npy',point_swm)   
        np.save(f'{out_path}/ind_FiberAnatMap_swm.npy',ind_FiberAnatMap_swm) 
        np.save(f'{out_path}/group_FiberAnatMap_swm.npy',group_FiberAnatMap_swm)

        i_dwm_lst = []
        # concatenate all 602 DWM clusters
        for j in range(198,800):
            index = np.where(test_predicted_lst==j)[0]
            # index = np.where(labels==i)[0]
            point_i = points[index]
            point_i = np.around(point_i,decimals=4)
            ind_FiberAnatMap = ind_FiberAnatMaps[index]
            if point_i.shape[0]!=0:
                i_dwm = j
                i_dwm_lst.append(i_dwm)
                if j == i_dwm_lst[0]:
                    point_dwm = point_i
                    ind_FiberAnatMap_dwm = ind_FiberAnatMap
                else:
                    point_dwm = np.concatenate((point_dwm,point_i))
                    ind_FiberAnatMap_dwm = np.concatenate((ind_FiberAnatMap_dwm,ind_FiberAnatMap))
        # Interpolate 
        repeats = ind_FiberAnatMap_dwm.shape[0] / 8000 +1 
        new_arr = np.repeat(atlas_cluster, repeats)
        if ind_FiberAnatMap_dwm.shape[0] >=8000:
            new_arr_ = np.random.choice(new_arr,ind_FiberAnatMap_dwm.shape[0] , replace=False)
        if ind_FiberAnatMap_dwm.shape[0] <8000:
            new_arr_ = np.random.choice(new_arr,ind_FiberAnatMap_dwm.shape[0] , replace=True)

        arrIndex = np.array(new_arr_).argsort()
        group_FiberAnatMap_dwm = new_arr_[arrIndex]  
        print('ind_FiberAnatMap_dwm',ind_FiberAnatMap_dwm.shape)
        print('group_FiberAnatMap_dwm',group_FiberAnatMap_dwm.shape)
        out_path = os.path.join(output_path,subject_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.save(f'{out_path}/point_dwm.npy',point_dwm)   
        np.save(f'{out_path}/ind_FiberAnatMap_dwm.npy',ind_FiberAnatMap_dwm) 
        np.save(f'{out_path}/group_FiberAnatMap_dwm.npy',group_FiberAnatMap_dwm)


def finetune_sequence(seg_cls,atlas,input_path,atlas_cluster,output_path):
    ''' '''
    
    paths = natural_sort(glob.glob(f'{input_path}/*/'))
    if seg_cls=='swm':
        k=2000
        # ind_FiberAnatMap_all = np.load(f'{input_path}/ind_FiberAnatMap_swm.npy')
        unique_atlas_cluster = np.unique(atlas_cluster)
        # normalize the ind_FiberAnatMap of WMA atlas
        nonzero_num = []
        atlas_new = np.zeros((atlas.shape))
        for i in range(2000): #TODO dwm是8000，swm是2000
            nonzero_index = np.nonzero(atlas[i])[0]  #返回非零元素的索引
            nonzero_num.append(nonzero_index.shape[0])
        # print(min(nonzero_num))  #8 表示2000swm_region中每行最少有8个非零值。后面就保留每一行的top8，其他的设为0
            top_k_idx = np.argpartition(atlas[i], -8)[-8:]
            # 将除了最大的8个数以外的元素设为0
            mask = np.zeros_like(atlas[i])
            mask[top_k_idx] = 1
            atlas_new[i] = atlas[i] * mask
        normalize_atlas_index_add_all = None
        for j in range(unique_atlas_cluster.shape[0]):
            index = np.where(atlas_cluster ==unique_atlas_cluster[j])[0]
            atlas_index = atlas_new[index]
            atlas_index_add = np.zeros((1,105))
            for x in range(atlas_index.shape[0]):  
                atlas_index_add+=atlas_index[x]
            normalize_atlas_index_add = atlas_index_add / (np.sum(atlas_index_add))
            if j == 0:
                normalize_atlas_index_add_all = normalize_atlas_index_add
            else:
                normalize_atlas_index_add_all = np.concatenate((normalize_atlas_index_add_all,normalize_atlas_index_add),axis=0)
        print('normalize_atlas_index_add_all',normalize_atlas_index_add_all.shape) 
        for i in range(len(paths)): 
            subject_name = paths[i].split('/')[-2]
            ind_FiberAnatMap_path = os.path.join(paths[i],'ind_FiberAnatMap_swm.npy')
            ind_FiberAnatMap =np.load(ind_FiberAnatMap_path)
            min_dis_index_lst = []
            for m in range(ind_FiberAnatMap.shape[0]):
                if np.nonzero(ind_FiberAnatMap[m])[0].shape[0]!=0:
                    test_region_cluster_dis = []
                    for n in range(8):  
                        B = ind_FiberAnatMap[m]/np.sum(ind_FiberAnatMap[m])
                        A = normalize_atlas_index_add_all[n]
                        cos_sim = (np.dot(A, B)) / (np.linalg.norm(A) * np.linalg.norm(B)) 
                        test_region_cluster_dis.append(cos_sim)
                else:
                    test_region_cluster_dis=[0]
                ##The most similar category is the revised one
                min_dis_index = np.argmax(np.array(test_region_cluster_dis)) 
                min_dis_index_lst.append(min_dis_index)
            out_path = os.path.join(output_path,subject_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            np.save(f'{out_path}/group_FiberAnatMap_swm_correction.npy',min_dis_index_lst)
    if seg_cls=='dwm':
        k=8000
        unique_atlas_cluster = np.unique(atlas_cluster)
        # normalize the ind_FiberAnatMap of WMA atlas
        nonzero_num = []
        atlas_new = np.zeros((atlas.shape))
        for i in range(8000): 
            nonzero_index = np.nonzero(atlas[i])[0]  
            nonzero_num.append(nonzero_index.shape[0])
        # print(min(nonzero_num))  #8 表示2000swm_region中每行最少有8个非零值。后面就保留每一行的top8，其他的设为0
            top_k_idx = np.argpartition(atlas[i], -8)[-8:]
            # 将除了最大的8个数以外的元素设为0
            mask = np.zeros_like(atlas[i])
            mask[top_k_idx] = 1
            atlas_new[i] = atlas[i] * mask
        normalize_atlas_index_add_all = None
        for j in range(unique_atlas_cluster.shape[0]):
            index = np.where(atlas_cluster ==unique_atlas_cluster[j])[0]
            atlas_index = atlas_new[index]
            atlas_index_add = np.zeros((1,105))
            for x in range(atlas_index.shape[0]):  
                atlas_index_add+=atlas_index[x]
            normalize_atlas_index_add = atlas_index_add / (np.sum(atlas_index_add))
            if j == 0:
                normalize_atlas_index_add_all = normalize_atlas_index_add
            else:
                normalize_atlas_index_add_all = np.concatenate((normalize_atlas_index_add_all,normalize_atlas_index_add),axis=0)
        print('normalize_atlas_index_add_all',normalize_atlas_index_add_all.shape) 
        for i in range(len(paths)): 
            subject_name = paths[i].split('/')[-2]
            ind_FiberAnatMap_path = os.path.join(paths[i],'ind_FiberAnatMap_dwm.npy')
            ind_FiberAnatMap =np.load(ind_FiberAnatMap_path)
            min_dis_index_lst = []
            for m in range(ind_FiberAnatMap.shape[0]):
                if np.nonzero(ind_FiberAnatMap[m])[0].shape[0]!=0:
                    test_region_cluster_dis = []
                    for n in range(34):  
                        B = ind_FiberAnatMap[m]/np.sum(ind_FiberAnatMap[m])
                        A = normalize_atlas_index_add_all[n]
                        cos_sim = (np.dot(A, B)) / (np.linalg.norm(A) * np.linalg.norm(B)) 
                        test_region_cluster_dis.append(cos_sim)
                else:
                    test_region_cluster_dis=[0]
                ##The most similar category is the revised one
                min_dis_index = np.argmax(np.array(test_region_cluster_dis)) 
                min_dis_index_lst.append(min_dis_index)
            out_path = os.path.join(output_path,subject_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            np.save(f'{out_path}/group_FiberAnatMap_dwm_correction.npy',min_dis_index_lst)
            path2 = os.path.join(out_path,'dwm_featMatrix.h5')
            point_dwm_path = os.path.join(paths[i],'point_dwm.npy')
            point_dwm =np.load(point_dwm_path)
            h5f = h5py.File(path2, 'w')
            h5f.create_dataset('point', data=point_dwm)
            h5f.create_dataset('ind_FiberAnatMap',data=ind_FiberAnatMap)
            h5f.create_dataset('group_FiberAnatMap',data=min_dis_index_lst)
        

if __name__ == '__main__':
    use_cpu = False
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    # the pretrained model weights in SupWMA
    weight_path = '../Pretrained_ModelWeights/TrainedModel_TwoStage/s1_cls'  
    parser = argparse.ArgumentParser(description="Classify SWM and DWM using the pretrained model in SupWMA.")
    parser.add_argument('--test_batch_size',type=int,default=512)
    parser.add_argument('--best_metric',type=str,default='f1')
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--output_path',type=str, default='../dataset/YourDataResults/',help='Output path.')
    parser.add_argument('--seg_cls', type=str, default='swm',help='The class which you want to segment further.')

    args = parser.parse_args()
    output_path = args.output_path
    best_metric = args.best_metric
    seg_cls = args.seg_cls
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    k = 800 #The number of classes we need to classify. 
    output_prediction_mask_path_all = []
    ## Inference
    print('******************Start inference on your data!**************************')
    paths = natural_sort(glob.glob(f'{output_path}/*/'))
    for i in range(len(paths)): 
        subject_name = paths[i].split('/')[-2]
        output_prediction_mask_path = os.path.join(output_path,'test_stage_one_mask_{}.h5'.format(i))
        output_prediction_mask_path_all.append(output_prediction_mask_path)
        if not os.path.exists(output_prediction_mask_path):
            test_dataset = SWMDWM_ClsDataset(num=i,input_path=output_path)
            test_data_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size = args.test_batch_size,shuffle=False,num_workers=int(args.num_workers))
            test_data_size = len(test_dataset)
            print('test data size : {}'.format(test_data_size))
            predicted_arr = test_net(k,weight_path,best_metric,test_data_loader,output_prediction_mask_path)
    # Process the results and fine tune the sequences
    if seg_cls=='swm':
        atlas_cluster = np.load('../WMA_atlas/swm_100_region_atlas_8cluster.npy')
        atlas = np.load('../WMA_atlas/swm_100_region_atlas.npy')

    if seg_cls=='dwm':
        atlas_cluster = np.load('../WMA_atlas/dwm_100_region_atlas_34cluster.npy')
        atlas = np.load('../WMA_atlas/dwm_100_region_atlas.npy')
    
    unique_atlas_cluster = np.unique(atlas_cluster)
    print('******************Save results!**************************')
    save_results(output_path,output_prediction_mask_path_all,output_path,atlas_cluster)
    print('******************Fine tune the sequence of group_FiberAnatMap!**************************')
    finetune_sequence(seg_cls,atlas,output_path,atlas_cluster,output_path)
    