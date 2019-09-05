# Example script to run the evaluation

#python -m smalst.smal_eval --name=smal_net_600 --img_path='smalst/zebra_testset/' --num_train_epoch=186 --use_annotations=True --mirror=True --segm_eval=True --img_ext='.jpg' --anno_path='smalst/zebra_testset/annotations'
python -m smalst.smal_eval --name=smal_net_600 --img_path='smalst/zebra_video_frame/' --num_train_epoch=186 --use_annotations=False --mirror=False --segm_eval=False --img_ext='.png' --bgval=0 --save_input=False --test_optimization_results=True --optimization_dir=smalst_experiments/demo_var_feat
