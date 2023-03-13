python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/imagenet/us_resnet50_simsiam_imagenet.yaml \
    --output /data/train_log_OFA/imagenet/r50/simsiam_momentum_us_r50_imagenet_sandwich3T_asymmertric_infoncev2_distill_new_head_A_200ep -j 8