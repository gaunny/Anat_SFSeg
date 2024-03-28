import vtk
import numpy as np
import re
import glob
import nibabel as nib
import os
from functools import reduce
from nibabel.affines import apply_affine
import argparse
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def downsample(input_path,output_path):
    ''' Downsample the whole brain fibers to an array with shape (10000,15,3).'''
    arender = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(arender)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    reader = vtk.vtkPolyDataReader()
    folders = natural_sort(glob.glob(f'{input_path}/*/'))
    for i in range(len(folders)):
        subject_name = folders[i].split('/')[-2]
        print('------------------------------------------------')
        print(folders[i])
        inputVTK = os.path.join(folders[i],'wb_fiber_reg_reg.vtk')
        reader.SetFileName(inputVTK)
        reader.Update()
        inpd = reader.GetOutput()
        inpoints = inpd.GetPoints()
        inpd.GetLines().InitTraversal()
        all_points = None
        for lidx in range(0, inpd.GetNumberOfLines()):
            ptids = vtk.vtkIdList()
            inpd.GetLines().GetNextCell(ptids)
            points = None
            for pidx in range(0, ptids.GetNumberOfIds()):
                point = np.array(inpoints.GetPoint(ptids.GetId(pidx)))
                if pidx == 0:
                    point = point[np.newaxis,:]
                    points = point
                else:
                    point = point[np.newaxis,:]
                    points = np.vstack((points,point))    
            ind = np.arange(ptids.GetNumberOfIds()) 
            # downsample each fiber to 15 points
            if ind.all() >14:
                sub_ind = np.random.choice(ind, 15, replace=False)
                print(sub_ind)
                arrIndex = np.array(sub_ind).argsort()  
                sub_ind_sort = sub_ind[arrIndex]
            else:
                sub_ind = np.random.choice(ind, 15, replace=True)
                arrIndex = np.array(sub_ind).argsort()
                sub_ind_sort = sub_ind[arrIndex]
            sub_points = np.array(points)[sub_ind_sort]
            if lidx == 0:
                sub_points = sub_points[np.newaxis,:]
                all_points = sub_points
            else:
                sub_points = sub_points[np.newaxis,:]
                all_points = np.vstack((all_points,sub_points))  
        
        sample = np.random.choice(all_points.shape[0],10000) # downsample fiber number to 10000
        down_features = None
        for k in range(len(sample)):
            down_feature = all_points[sample[k],:,:].reshape(1,15,3)  
            if k == 0:
                down_features = down_feature
            else:
                down_features = np.vstack((down_features,down_feature))  
        print('down_features',down_features.shape) #(10000,15,3)
        save_path = os.path.join(output_path,subject_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(f'{save_path}/downsampled.npy',down_features)


def fiber2volume(input_path,output_path):
    ''' Apply affine transformation to map the fibers on the parcellation images.'''
    out_folders = natural_sort(glob.glob(f'{output_path}/*/'))
    for m in range(len(out_folders)):  
        subject_name = out_folders[m].split('/')[-2]
        print('------------------------------------------------')
        print(out_folders[m])
        points_file = os.path.join(out_folders[m],'downsampled.npy')
        down_features = np.load(points_file)
        refvolume = os.path.join(input_path,subject_name,'aparc+aseg.nii.gz')
        volume = nib.load(refvolume)
        volume_shape = volume.get_fdata().shape 
        volume_content = volume.get_fdata().astype(int)
        label_set = set(np.unique(volume_content))
        label_list = list(map(int,label_set))
        len_label_list = len(label_list)
        ind_FiberAnatMap_original = None
        # apply affine transformation to map the fiber points on the parcellation image
        point_ijk = apply_affine(np.linalg.inv(volume.affine), down_features)
        point_ijk = np.rint(point_ijk).astype(np.int32)
        regions = []
        for n in range(down_features.shape[0]): #10000
            point_list = [(point_ijk[n,j,0], point_ijk[n,j,1], point_ijk[n,j,2]) for j in range(point_ijk.shape[1])]
            point_list = set(point_list)
            region = np.zeros((len_label_list,), dtype=int)
            for x, y, z in list(point_list):
                c_idx = volume_content[x, y, z]
                c_idx_ = label_list.index(c_idx)
                region[c_idx_] += 1
            regions.append(region)
            fiber2region = np.vstack((region,label_list))
            if n == 0:
                fiber2region = fiber2region[np.newaxis,:]
                ind_FiberAnatMap_original = fiber2region
            else:
                fiber2region = fiber2region[np.newaxis,:]
                ind_FiberAnatMap_original = np.vstack((ind_FiberAnatMap_original,fiber2region))
        print('ind_FiberAnatMap_original',ind_FiberAnatMap_original.shape) #[10000,2,110]
        save_path = os.path.join(output_path,subject_name,'ind_FiberAnatMap_original.npy')
        np.save(save_path,ind_FiberAnatMap_original)

def delete_wm(output_path):
    ''' Delete unnecessary brain regions from 'ind_FiberAnatMap_original.npy',
     like white matter (left and right hemisphere), WM-hypointensities , background(0) and CSF.
     Make sure the final dimension of ind_FiberAnatMap is 105. (105 cortical and subcortical regions totally).'''
    sort_k_label_list = []
    content_list = []
    folders = natural_sort(glob.glob(f'{output_path}/*/'))
    for i in range(len(folders)): 
        subject_name = folders[i].split('/')[-2]
        print('------------------------------------------------')
        print(folders[i])
        ind_FiberAnatMap_original_file = os.path.join(folders[i],'ind_FiberAnatMap_original.npy')
        ind_FiberAnatMap_original = np.load(ind_FiberAnatMap_original_file)
        arrIndex = np.array(ind_FiberAnatMap_original[0,1,:]).argsort()
        sort_k_label = list(ind_FiberAnatMap_original[0,1,:][arrIndex])
        sort_k_label_list.append(sort_k_label)
        content_list.append(ind_FiberAnatMap_original)
    # Compute the intersection of all subjects' brain regions classes in ind_FiberAnatMap_original
    intersect = reduce(np.intersect1d,[sort_k_label_list[k] for k in range(len(sort_k_label_list))])
    intersect = list(intersect)
    print('intersect_original',len(intersect))
    if len(intersect)==108:
        intersect.remove(0)  #background
        intersect.remove(2)  #wm
        intersect.remove(41)  #wm
    if len(intersect)==109:
        intersect.remove(0)  #background
        intersect.remove(2)  #wm
        intersect.remove(41)  #wm 
        intersect.remove(77) #WMH
    if len(intersect)==110:
        intersect.remove(0)  #background
        intersect.remove(2)  #wm
        intersect.remove(41)  #wm 
        intersect.remove(77) #WMH
        intersect.remove(24) #CSF
    if len(intersect)==111:
        intersect.remove(0)  #background
        intersect.remove(2)  #wm
        intersect.remove(41)  #wm 
        intersect.remove(77) #WMH
        intersect.remove(24) #CSF
        intersect.remove(80) #non-WM-hypointensities    
    print('intersect_deleted:',len(intersect)) #105
    #把num和新的re拼起来
    ind_FiberAnatMap_all = None
    ind_FiberAnatMap_all_subjects = None
    for j in range(len(folders)):
        for k in range(len(intersect)):
            index = np.where(content_list[j][0,1,:]==intersect[k])
            ind_FiberAnatMap = content_list[j][:,0,index]
            label = content_list[j][:,1,index]
            ind_FiberAnatMap_ = np.concatenate((ind_FiberAnatMap,label),axis=1)
            if k == 0:
                ind_FiberAnatMap_all = ind_FiberAnatMap_
            else:
                ind_FiberAnatMap_all = np.concatenate((ind_FiberAnatMap_all,ind_FiberAnatMap_),axis=2)
        print('ind_FiberAnatMap_all',ind_FiberAnatMap_all.shape)  #(10000,2,105)
        save_path = os.path.join(folders[j],'ind_FiberAnatMap.npy')
        np.save(save_path,ind_FiberAnatMap_original)

        if j == 0 :
            ind_FiberAnatMap_all = ind_FiberAnatMap_all[np.newaxis,:]
            ind_FiberAnatMap_all_subjects = ind_FiberAnatMap_all
        else:
            ind_FiberAnatMap_all = ind_FiberAnatMap_all[np.newaxis,:]
            ind_FiberAnatMap_all_subjects = np.vstack((ind_FiberAnatMap_all,ind_FiberAnatMap_all_subjects))
    print('ind_FiberAnatMap_all_subjects',ind_FiberAnatMap_all_subjects.shape) #(number_of_subjects,10000,2,105)
    #save all subjects' ind_FiberAnatMap
    save_path = os.path.join(output_path,'ind_FiberAnatMap_all_subjects.npy')
    np.save(save_path,ind_FiberAnatMap_all_subjects)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input_path', default='../dataset/YourData',
                        help='This folder must contain each subfolder of the test subjects, and each subfolder contains ''wb_fiber_reg_reg.vtk'' and ''aparc+aseg.nii.gz''.')
    parser.add_argument('--output_path', default='../dataset/YourDataResults',help='The output folder.')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('******************Start downsampling!**************************')
    downsample(input_path,output_path)
    print('******************Start affine transformation!**************************')
    fiber2volume(input_path,output_path)
    print('******************Start delete unnecessary regions!**************************')
    delete_wm(output_path)