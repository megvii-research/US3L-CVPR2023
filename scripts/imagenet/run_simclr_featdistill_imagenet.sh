python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/imagenet/us_resnet50_simclr_featdistill_imagenet.yaml \
    --output /data/train_log_OFA/imagenet/r50/simclr_momenrtum_max_with_momentum_r50_imagenet_sandwich3T_asymmertric_mse_distill_feat_2_3_4_distill_mse_groupreg_A -j 8
