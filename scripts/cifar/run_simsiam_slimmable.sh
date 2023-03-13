python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/s_resnet18_simsiam_cifar.yaml \
    --output /data/train_log_OFA/cifar100/r18/simsiam_s_r18_cifar100_1.0_0.5_mse_distill -j 8