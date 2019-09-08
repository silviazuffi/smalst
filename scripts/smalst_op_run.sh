#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib

# The directory processed contains the input image as processed by the network in the feed forward pass.
# Before running the optimization please run the feed forward network on the data with the flag save_input at True (see smalst_evaluation_run.sh)

zebra_dir='/Users/silvia/Dropbox/Work/smalst/zebra_video_frame/processed'
zebra_dir_images='/Users/silvia/Dropbox/Work/smalst/zebra_video_frame/processed/*'
for dir in $zebra_dir_images
do
    for fil in $dir
    do
        file=$(basename $fil)
        echo "$file"
        python -m smalst.experiments.smal_shape --name=smal_net_600 --zebra_dir=$zebra_dir --image_file_string=$fil --num_pretrain_epochs=186 --batch_size=1 --texture_map=False --perturb_bbox=False --save_epoch_freq=1000 --save_training_imgs=True --is_optimization=True --use_loss_on_whole_image=True --learning_rate=0.0001 --use_directional_light=False --num_epochs=220 --is_var_opt=False
    done
done

