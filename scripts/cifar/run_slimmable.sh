#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#    slimmable  --config ./configs/cifar/resnet18_slimmable_cifar.yaml \
#    --output /data/train_log_OFA/resnet18_slimmable_distill_cifar100 -j 8

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    slimmable  --config ./configs/cifar/mobilenetv2_slimmable_cifar.yaml \
    --output /data/train_log_OFA/mobilenetv2_slimmable_distill_cifar100 -j 8