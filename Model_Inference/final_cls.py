from load_testdata import SegDataset
import torch
import argparse
import os
import time
import h5py
import numpy as np
import re
import glob
import vtk
import sys
sys.path.append("..")
from model import SegNet
use_cpu = True
if use_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_test_data(seg_cls):
    test_dataset = SegNet(seg_cls)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = args.test_batch_size,shuffle=False,num_workers=int(args.num_workers))
    test_data_size = len(test_dataset)
    print('test data size : {}'.format(test_data_size))
    # num_classes = len(test_dataset.label_names)
    num_classes = k
    # lable_names = test_dataset.label_names
    return test_loader, num_classes

def load_model(seg_cls,weight_path):
    classifer = SegNet(seg_cls,k=k).to(device)
    classifer_weight_path = os.path.join(weight_path,'best_{}_model.pth'.format(args.best_metric))
    classifer.load_state_dict(torch.load(classifer_weight_path))
    return classifer

def test_net(seg_cls,weight_path,test_data_loader,output_prediction_mask_path):
    print('')
    print('===================================')
    print('')
    classifer_net = load_model(seg_cls,weight_path)
    start_time = time.time()
    with torch.no_grad():
        test_predicted_lst = []
        for j,data in (enumerate(test_data_loader, 0)):
            points,group_FiberAnatMap,ind_FiberAnatMap = data
            # print('group_FiberAnatMap',group_FiberAnatMap.shape)
            # print('ind_FiberAnatMap',ind_FiberAnatMap.shape)
            points = points.transpose(2, 1)  
            points,group_FiberAnatMap,ind_FiberAnatMap = points.to(device),group_FiberAnatMap.to(device),ind_FiberAnatMap.to(device)
            classifer_net = classifer_net.eval()
            pred = classifer_net(points,group_FiberAnatMap,ind_FiberAnatMap)
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

def save_results(input_path,output_prediction_mask_path_all,output_path,seg_cls):
    paths = natural_sort(glob.glob(f'{input_path}/*/'))
    for m in range(len(paths)):
        subject_name = paths[m].split('/')[-2]
        if seg_cls=='swm':
            vtk_path = os.path.join(output_path,subject_name,'swm_results')
            num_class = 198
            if not os.path.exists(vtk_path):
                os.makedirs(vtk_path)
            point_file = os.path.join(paths[m],'move_outlier_swm.h5')
        if seg_cls=='dwm':
            num_class = 800
            vtk_path = os.path.join(output_path,subject_name,'dwm_results')
            if not os.path.exists(vtk_path):
                os.makedirs(vtk_path)
            point_file = os.path.join(paths[m],'dwm_featMatrix.h5')
        feat = h5py.File(point_file,'r')
        points_feat = np.array(feat['point'])
        h5_path = output_prediction_mask_path_all[m]
        h5_feat = h5py.File(h5_path,'r')
        test_predicted_lst = np.asarray(h5_feat['complete_pred_test'])
        # Write to VTK
        for k in range(num_class):
            index = np.where(test_predicted_lst==k)[0]
            point_i = points_feat[index]
            point_i = np.around(point_i,decimals=4)
            lines = vtk.vtkCellArray()
            points = vtk.vtkPoints()
            for i in range(point_i.shape[0]):
                line = vtk.vtkPolyLine()
                for j in range(point_i.shape[1]):
                    point_id = points.InsertNextPoint(point_i[i, j])
                    line.GetPointIds().InsertNextId(point_id)
                lines.InsertNextCell(line)  
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            writer = vtk.vtkPolyDataWriter()
            
            writer.SetFileName(os.path.join(vtk_path,'cluster_{}_{}.vtk'.format(seg_cls,k)))
            writer.SetInputData(polydata)
            writer.SetFileTypeToBinary()
            writer.SetFileVersion(42)
            writer.Write()

if __name__ == "__main__":
    use_cpu = False
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Segment SWM fiber bundles.")

    parser.add_argument('--weight_path', type=str,default='../Pretrained_ModelWeights/TrainedModel_AnatSFSeg/swm_100/5/', help='pretrained network model')
    parser.add_argument('--output_path', type=str,default='../dataset/YourDataResults/', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--test_batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--best_metric', type=str, default='f1', help='evaluation metric')
    parser.add_argument('--supcon_epoch', type=int, default=100, help='The epoch of encoder model')
    parser.add_argument('--seg_cls', type=str, default='swm',help='The class which you want to segment further.')

    args = parser.parse_args()
    output_path = args.output_path
    seg_cls = args.seg_cls
    weight_path = args.weight_path
    if seg_cls=='swm':
        k=198  #SWM class
        weight_path = '../Pretrained_ModelWeights/TrainedModel_AnatSFSeg/swm_100/5/'
    if seg_cls=='dwm':
        k=800  #DWM + SWM outliers class
        weight_path = '../Pretrained_ModelWeights/TrainedModel_AnatSFSeg/dwm_100/4/'
    ## Inference
    print('******************Start Inference!**************************')
    output_prediction_mask_path_all = []
    paths = natural_sort(glob.glob(f'{output_path}/*/'))
    for i in range(len(paths)): 
        subject_name = paths[i].split('/')[-2]
        if seg_cls=='swm':
           output_prediction_mask_path = os.path.join(output_path, 'test_swmseg_mask_{}.h5'.format(i))
        if seg_cls=='dwm':
            output_prediction_mask_path = os.path.join(output_path, 'test_dwmseg_mask_{}.h5'.format(i))
        output_prediction_mask_path_all.append(output_prediction_mask_path)
        test_dataset = SegDataset(num=i,input_path=output_path,seg_cls=seg_cls)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size = args.test_batch_size,shuffle=False,num_workers=int(args.num_workers))
        test_data_size = len(test_dataset)
        print('test data size : {}'.format(test_data_size))
        predicted_arr = test_net(seg_cls,weight_path,test_data_loader,output_prediction_mask_path)
     # Process the results
    print('******************Save results!**************************')
    
    save_results(output_path,output_prediction_mask_path_all,output_path,seg_cls)