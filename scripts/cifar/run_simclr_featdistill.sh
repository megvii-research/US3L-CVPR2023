#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
#    ssl --config ./configs/cifar/us_resnet50_simclr_featdistill_cifar.yaml \
#    --output /data/train_log_OFA/cifar100/r50/infonce_bp_ablation/group_reg+featdistill -j 8

#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#    ssl --config ./configs/cifar/us_mobilenetv2_simclr_featdistill_cifar.yaml \
#    --output /data/train_log_OFA/cifar100/mbv2/infonce_bp_ablation/group_reg_only -j 8


python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/us_resnet18_simclr_featdistill_cifar.yaml \
    --output ./train_log_OFA/cifar100/r18/ours_base_nce_distill_mse_400ep -j 8
