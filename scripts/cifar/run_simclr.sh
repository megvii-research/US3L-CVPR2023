python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/us_resnet18_simclr_cifar.yaml \
    --output /data/train_log_OFA/cifar100/r18/simclr_base_byol_us_r18_cifar100_sandwich3T_mse_distill -j 8