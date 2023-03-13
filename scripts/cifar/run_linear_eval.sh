python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/cifar/resnet18_linear_eval_cifar.yaml \
    --output ./train_log_OFA/resnet18_linear_eval_cifar100 -j 8

#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#    linear  --config ./configs/cifar/resnet50_linear_eval_cifar.yaml \
#    --output /data/train_log_OFA/resnet50_linear_eval_cifar100 -j 8

#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#    linear  --config ./configs/cifar/mobilenetv2_linear_eval_cifar.yaml \
#    --output /data/train_log_OFA/mobilenetv2_linear_eval_cifar100 -j 8