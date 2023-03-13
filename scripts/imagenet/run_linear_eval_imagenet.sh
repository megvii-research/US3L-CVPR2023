python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/imagenet/resnet50_linear_eval_imagenet.yaml \
    --output /data/train_log_OFA/resnet50_linear_eval_imagene_0.75_baseline -j 8