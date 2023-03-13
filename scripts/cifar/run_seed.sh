python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    distill --config ./configs/cifar/us_resnet18_seed_simclr_cifar.yaml \
    --output /data/train_log_OFA/cifar100/r50/seed_simclr_us_r50_cifar100_0.25_teacher_byol_r50 -j 8