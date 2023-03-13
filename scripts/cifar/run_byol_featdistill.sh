python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/us_resnet18_byol_featdistill_cifar.yaml \
    --output ./train_log_OFA/cifar100/r18/ours_base_mse_distill_mse_400ep -j 4

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
#    ssl --config ./configs/cifar/us_resnet50_byol_featdistill_cifar.yaml \
#    --output /data/train_log_OFA/cifar100/r50/byol_us_r50_cifar100_sandwichOnce3T_mse_distill_feat_2.2_3.5_4.2_distill_mse_groupreg_linear_lambda_2.5e-4_alpha_0.05_dynamic_400_bins_8_warmup_200_A_800ep -j 8


#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#    ssl --config ./configs/cifar/us_mobilenetv2_byol_featdistill_cifar.yaml \
#    --output /data/train_log_OFA/cifar100/mbv2/byol_us_mobilenetv2_cifar100_sandwichOnce3T_mse_distill_groupreg_linear_lambda_2.5e-4_alpha_0.05_dynamic_400_bins_8_warmup_200_A_800ep -j 8