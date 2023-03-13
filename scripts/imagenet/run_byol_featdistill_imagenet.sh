#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#    ssl --config ./configs/imagenet/us_resnet18_byol_featdistill_imagenet.yaml \
#    --output ./train_log_OFA/imagenet/r18/simsiam_momentum_us_r18_imagenet_sandwich3T_asymmertric_infoncev2_distill_new_head_200ep -j 8

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/imagenet/us_resnet50_byol_featdistill_imagenet.yaml \
    --output ./train_log_OFA/imagenet/r50/ours_base_mse_distill_mse_100ep -j 8

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
#    ssl --config ./configs/imagenet/us_mobilenetv2_byol_featdistill_imagenet.yaml \
#    --output /data/train_log_OFA/imagenet/mbv2/simsiam_momentum_us_mbv2_imagenet_sandwich3T_asymmertric_infoncev2_distill_A_new_head_200ep -j 8