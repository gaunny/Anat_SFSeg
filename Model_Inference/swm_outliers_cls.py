import numpy as np
import argparse
import h5py
import time
import os
import pickle
import re
import glob
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from swm_outliers_model import PointNet_SupCon, PointNet_Classifier
from load_testdata import SWMOutliers_ClsDataset

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_model():
    """load stage 1 and stage 2 models"""
    stage2_encoder = PointNet_SupCon(head=encoder_params['head_name'], feat_dim=encoder_params['encoder_feat_num']).to(device)
    stage2_classifer = PointNet_Classifier(num_classes=encoder_params['stage2_num_class']).to(device)
    # load weights
    encoder_weight_path = os.path.join(args.weight_path, 's2_encoder', 'epoch_{}_model.pth'.format(args.supcon_epoch))
    stage2_encoder.load_state_dict(torch.load(encoder_weight_path))
    classifier_weight_path = os.path.join(args.weight_path, 's2_cls', 'best_{}_model.pth'.format(args.best_metric))
    stage2_classifer.load_state_dict(torch.load(classifier_weight_path))
    return stage2_encoder, stage2_classifer


def test_net(output_prediction_mask_path):
    """perform predition of multiple clusters"""
    print('')
    print('===================================')
    print('')
    # output_prediction_mask_path = os.path.join(args.weight_path+ 'test_prediction_mask_2_MC001.h5')
    stage2_encoder_net, stage2_classifer_net = load_model()
    if not os.path.exists(output_prediction_mask_path):
        # Load model
        start_time = time.time()
        with torch.no_grad():
            test_predicted_lst = []
            for j, data in (enumerate(test_data_loader, 0)):
                # stage 2
                swm_points=data
                swm_points = swm_points.transpose(2, 1)
                swm_points = swm_points.to(device)
                stage2_encoder_net, stage2_classifer_net = \
                    stage2_encoder_net.eval(), stage2_classifer_net.eval()
                features = stage2_encoder_net.encoder(swm_points)
                stage2_pred = stage2_classifer_net(features)
                _, stage2_pred_idx = torch.max(stage2_pred, dim=1)
                stage2_pred_idx = torch.where(stage2_pred_idx < k, stage2_pred_idx, torch.tensor(k).to(device))
                
                stage2_pred_idx = stage2_pred_idx.cpu().detach().numpy().tolist()
                test_predicted_lst.extend(stage2_pred_idx)
        end_time = time.time()
        print('The total time of prediction is:{} s'.format(round((end_time - start_time), 4)))
        print('The test sample size is: ', len(test_predicted_lst))
        
        test_prediction_lst_h5 = h5py.File(output_prediction_mask_path, "w")
        test_prediction_lst_h5['complete_pred_test'] = test_predicted_lst
        test_predicted_array = np.asarray(test_predicted_lst)

    else:
        print('Loading prediction result.')
        test_prediction_h5 = h5py.File(output_prediction_mask_path, "r")
        test_predicted_array = np.asarray(test_prediction_h5['complete_pred_test'])

    return test_predicted_array

def save_results(input_path,output_prediction_mask_path_all):
    paths = natural_sort(glob.glob(f'{input_path}/*/'))
    for m in range(len(paths)):
        subject_name = paths[m].split('/')[-2]
        points_file = os.path.join(paths[m],'point_swm.npy')
        ind_FiberAnatMap_file = os.path.join(paths[m],'ind_FiberAnatMap_swm.npy')
        group_FiberAnatMap_file = os.path.join(paths[m],'group_FiberAnatMap_swm_correction.npy')
        h5_path = output_prediction_mask_path_all[m]
        print('h5_path',h5_path)
        h5_feat = h5py.File(h5_path,'r')
        test_predicted_lst = np.asarray(h5_feat['complete_pred_test'])
        print('test_predicted_lst',test_predicted_lst.shape)
        points = np.load(points_file)
        print('points',points.shape)
        ind_FiberAnatMap = np.load(ind_FiberAnatMap_file)
        group_FiberAnatMap = np.load(group_FiberAnatMap_file)
        point_all = None
        ind_FiberAnatMap_all = None
        group_FiberAnatMap_all = None
        for k in range(198):
            index = np.where(test_predicted_lst==k)[0]
            point_ = points[index]
            group_FiberAnatMap_ = group_FiberAnatMap[index]   
            ind_FiberAnatMap_ = ind_FiberAnatMap[index]
            if k ==0:
                point_all = point_
                ind_FiberAnatMap_all = ind_FiberAnatMap_
                group_FiberAnatMap_all = group_FiberAnatMap_
            else:
                point_all = np.concatenate((point_all,point_),axis=0)
                ind_FiberAnatMap_all =np.concatenate((ind_FiberAnatMap_all,ind_FiberAnatMap_),axis=0)
                group_FiberAnatMap_all = np.concatenate((group_FiberAnatMap_all,group_FiberAnatMap_),axis=0)
        path2 = os.path.join(paths[m],'move_outlier_swm.h5')
        h5f = h5py.File(path2, 'w')
        h5f.create_dataset('point', data=point_all)
        h5f.create_dataset('ind_FiberAnatMap',data=ind_FiberAnatMap_all)
        h5f.create_dataset('group_FiberAnatMap',data=group_FiberAnatMap_all)


if __name__ == "__main__":
    use_cpu = False
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    k=198  #SWM outliers class
    out_path = '/data/home/dzhang/data1/AD_MCI/fiber2region_result/Anat_SFSeg_output/AD05/swm_outliers/'  
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Use the stage-two model of SupWMA to filter the swm outliers.")
    parser.add_argument('--weight_path', type=str,default='../Pretrained_ModelWeights/TrainedModel_TwoStage/', help='pretrained network model')
    parser.add_argument('--output_path', type=str,default='../dataset/YourDataResults/', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--test_batch_size', type=int, default=512, help='batch size')

    parser.add_argument('--best_metric', type=str, default='f1', help='evaluation metric')
    parser.add_argument('--supcon_epoch', type=int, default=100, help='The epoch of encoder model')

    args = parser.parse_args()
    output_path = args.output_path

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.weight_path, 's1_cls', 'stage1_params.pickle'), 'rb') as f:
        stage1_params = pickle.load(f)
        f.close()
    with open(os.path.join(args.weight_path, 's2_encoder', 'encoder_params.pickle'), 'rb') as f:
        encoder_params = pickle.load(f)
        f.close()
    output_prediction_mask_path_all = []
    ## Inference
    print('******************Start inference on your data!**************************')
    paths = natural_sort(glob.glob(f'{output_path}/*/'))
    for i in range(len(paths)): 
        subject_name = paths[i].split('/')[-2]
        output_prediction_mask_path = os.path.join(output_path,'test_swmoutliers_mask_{}.h5'.format(i))
        output_prediction_mask_path_all.append(output_prediction_mask_path)
        if not os.path.exists(output_prediction_mask_path):
            test_dataset = SWMOutliers_ClsDataset(num=i,input_path=output_path)
            test_data_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size = args.test_batch_size,shuffle=False,num_workers=int(args.num_workers))

            test_data_size = len(test_dataset)
            print('test data size : {}'.format(test_data_size))
            predicted_arr = test_net(output_prediction_mask_path)
    # Process the results
    print('******************Save results!**************************')
    save_results(output_path,output_prediction_mask_path_all)


