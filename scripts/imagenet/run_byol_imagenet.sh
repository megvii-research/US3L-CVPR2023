python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/imagenet/us_resnet50_byol_imagenet.yaml \
    --output /data/train_log_OFA/imagenet/r50/simsiam_us_r50_imagenet_100ep_single_0.5 -j 8
