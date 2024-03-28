#!/bin/bash
# Prepare your datasets
####################################################################################
## input_path' contains the whole brain tractogram vtk files
## Please make sure that the vtk file is registered in the ORG atlas first. If not, please use the command below:
## Example:
wm_register_to_atlas_new.py -l 40 -mode affine ../dataset/YourData/sub_01/wb_fiber.vtk ../WMA_atlas/ORG-Atlases-1.1.1/ORG-800FC-100HCP/atlas.vtp ../dataset/YourData/sub_01/fiberregistration
wm_register_to_atlas_new.py -mode nonrigid ../dataset/YourData/sub_01/fiberregistration/wb_fiber/output_tractography/wb_fiber_reg.vtk ../WMA_atlas/ORG-Atlases-1.1.1/ORG-800FC-100HCP/atlas.vtp ../dataset/YourData/sub_01/fiberregistrationnonrigid
cp ../fiberregistrationnonrigid/wb_fiber_reg/output_tractography/wb_fiber_reg_reg.vtk ../dataset/YourData/sub_01/

####################################################################################
## Please make sure that T1 image has processed through 'reconall' 
## the cortical and subcortical file -- aparc+aseg.mgz is converted to aparc_aseg.nii
## If not, use the command below:
mrconvert ../dataset/YourData/sub_01/reconall_results/aparc+aseg.mgz ../dataset/YourData/sub_01/aparc+aseg.nii.gz

####################################################################################
## prepare the test datasets:
seg_cls=swm
input_path=../dataset/YourData
output_path=../dataset/YourDataResults

cd ..
cd Model_Inference
echo "**************************************************************"
echo "Prepare the test dataset!"
echo "**************************************************************"
python prepare_testdata.py --input_path ${input_path} --output_path ${output_path}

## Firstly, we need to classify the whole brain fibers into SWM and DWM two parts using the stage-one pretrained model of SupWMA
## Please also cite SupWMA: Xue T, Zhang F, Zhang C, et al. Superficial white matter analysis: An efficient point-cloud-based deep learning framework with supervised contrastive learning for consistent tractography parcellation across populations and dMRI acquisitions[J]. Medical Image Analysis, 2023, 85: 102759.
echo "**************************************************************"
echo "Classify SWM and DWM first!"
echo "**************************************************************"
python stage_one_cls.py --seg_cls ${seg_cls} --output_path ${output_path}
echo "**************************************************************"
echo "Filter SWM outliers!"
echo "**************************************************************"
python swm_outliers_cls.py --output_path ${output_path}
echo "**************************************************************"
echo "Segment fibers finely!"
echo "**************************************************************"
python final_cls.py --seg_cls ${seg_cls} --output_path ${output_path}