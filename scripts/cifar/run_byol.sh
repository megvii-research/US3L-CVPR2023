python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/us_resnet18_byol_cifar.yaml \
    --output /data/train_log_OFA/cifar100/r18/byol_us_r18_cifar100_sandwich_mse_distill_m_0.0 -j 8