# Anat-SFSeg
Code for our *Medical Image Analysis* paper "Anat-SFSeg: Anatomically-guided superﬁcial ﬁber segmentation with point-cloud deep learning"
![framework](img/pipeline_revision.png)
If you find this code useful in your research please cite

## Setup
The main environment is:
- cuda 11.3
- torch 1.11.0
- nibabel 4.0.2
- whitematteranalysis 0.4.0 (``` pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git ```)
- h5py 3.7.0
- vtk 9.2.6
  
## Train Model
``` 
cd scripts
bash train.sh 
```

## Inference on your data
You need to apply tractography on your diffusion data to obtain the whole brain tractogram. And you need to apply cortical and subcortical parcellation *(reconall in FreeSurfer)* on your T1w data to obtain the parcellation result *(aparc+aseg.mgz)*.
For example, we provide two subjects' tractograms and parcellations in ``` dataset/YourData ```. Here we use the unscented Kalman filter (UKF) tractography method to obtain the whole brain tractogram named 'wb_fiber.vtk', and the cortical and subcortical parcellation images 'aparc+aseg.nii.gz' are also provided.
``` 
cd scripts
bash test.sh 
```
## The results analysis
The fiber bundle category to which the obtained clusters belong are listed in table ```swm_labels.csv``` and ```dwm_labels.csv```. Note that the ```labels``` in the table is the last number of the resulting bundle (.vtk) file.

## References
Thanks to the code of SupWMA (https://github.com/SlicerDMRI/SupWMA), this is the project we rely on.
Please cite the following papers for using the code and/or the training data:
``` 
Xue T, Zhang F, Zhang C, et al. Superficial white matter analysis: An efficient point-cloud-based deep learning framework with supervised contrastive learning for consistent tractography parcellation across populations and dMRI acquisitions[J]. Medical Image Analysis, 2023, 85: 102759.

Zhang, F., Wu, Y., Norton, I., Rathi, Y., Makris, N., O'Donnell, LJ. 
An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. 
NeuroImage, 2018 (179): 429-447

James G Malcolm, Martha E Shenton, Yogesh Rathi (2009). Neural tractography using an unscented Kalman filter. In International Conference on Information Processing in Medical Imaging, pp. 126–138.

O'Donnell LJ, Wells III WM, Golby AJ, Westin CF. 
Unbiased groupwise registration of white matter tractography.
In MICCAI, 2012, pp. 123-130.

```

