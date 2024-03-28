#!/bin/bash
# Train SWM and DWM segmentation model using WMA datasets respectively
seg_cls=dwm  #select from 'swm' or 'dwm' 
input_path=../dataset/WMA/
output_weights=../ModelWeights/

python ../train.py \
    --seg_cls ${seg_cls} \
    --input_path ${input_path} \
    --output_weights ${output_weights} \
    --epoch 200 \
    --batch_size 1024
    
