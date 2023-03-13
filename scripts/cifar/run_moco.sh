python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/resnet50_moco_cifar.yaml \
    --output /data/train_log/cifar100/r50/mocov2_cifar_r50_cifar100_baseline -j 16