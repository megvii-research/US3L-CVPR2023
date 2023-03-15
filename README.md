# US3L-CVPR2023
Official code for Three Guidelines You Should Know for Universally Slimmable Self-Supervised Learning (Accepted to **CVPR2023**).


## Introduction
We propose universally slimmable self-supervised learning (dubbed as US3L) to achieve better accuracy-efficiency trade-offs for deploying self-supervised models across different devices.

## Getting Started

### Prerequisites
* Python 3
* PyTorch (= 1.10.0)
* Torchvision (= 0.11.1)
* Numpy
* CUDA 10.1

All dataset definitions are in the [datasets](datasets/) folder. By default, the form of PyTorch's train/val folder is used. You can specify the path of the dataset yourself in the corresponding dataset file.

We used 4 2080Ti GPUs for CIFAR experiments (except for ResNet-50) and 8 2080Ti GPUs for ImageNet experiments.

### CIFAR Experiments
- Pre-training stage using US3L (c.f. [scripts/cifar/run_simclr_featdistill.sh](scripts/cifar/run_simclr_featdistill.sh)), run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/us_resnet18_simclr_featdistill_cifar.yaml \
    --output [your_checkpoint_dir] -j 8
```
You can set hyper-parameters manually in the corresponding .yaml file (e.g., [configs/cifar/us_resnet18_simclr_featdistill_cifar.yaml](configs/cifar/us_resnet18_simclr_featdistill_cifar.yaml) here).
```python
MODEL:
    ARCH: us_cifar_resnet18 # backbone model
    OFA:
        WIDTH_MULT_RANGE: [0.25, 1.0]  # range of widths
        NUM_SAMPLE_TRAINING: 3 # number of sampled sub-networks
        SANDWICH: False # whether to use the sandwich rule
        SAMPLE_SCHEDULER: A # A denotes dynamic sampling, set it to "" to deactivate
        MOMENTUM_UPDATE: False # False: only update momentum encoder once (max model forward pass)
        DISTILL: True # whether to use inplace distillation
        DISTILL_CRITERION: CosineSimilarity # distillation loss
        REGULARIZER: True # whether to use group regularization
        REGULARIZER_CRITERION: GroupReg
        REGULARIZER_LAMBDA: 0.00005 # set it to 1/2 weight decay
        REGULARIZER_DECAY_ALPHA: 0.05 # hyper-parameter alpha
        REGULARIZER_DECAY_TYPE: linear # schedule
        REGULARIZER_DECAY_BINS: 8
TRAIN:
    EPOCHS: 400 # number of epochs
    DATASET: cifar100 # dataset
    BATCH_SIZE: 128 # per-gpu batch size
    LOSS:
        CRITERION:
            NAME: InfoNCEv2_distill # base loss
SSL:
    TYPE: SimCLR_Momentum # SSL algorithm
    SETTING:
        DIM: 256
        HIDDEN_DIM: 2048
        T: 0.5
        MLP: TRUE
        PREDICTOR: True # use auxillary distillation head
        MOMENTUM: 0.99
```

- You can also use other baseline methods for pre-training. For example, you can run [scripts/cifar/run_simsiam.sh](scripts/cifar/run_simsiam.sh) for SimSiam, [scripts/cifar/run_moco.sh](scripts/cifar/run_moco.sh) for MoCo.

- Linear evaluation stage using pre-trained models (c.f. [scripts/cifar/run_linear_eval.sh](scripts/cifar/run_linear_eval.sh)), run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/cifar/resnet18_linear_eval_cifar.yaml \
    --output [output_dir] -j 8
```
To specify the pre-trained model path and evaluation channel-width, you need to mannually modify the .yaml file ([configs/cifar/resnet18_linear_eval_cifar.yaml](./configs/cifar/resnet18_linear_eval_cifar.yaml)). For instance, our pretrained model is ./checkpoint.pth.tar and we want to evaluate it with channel width at 0.25x, the modification in yaml file should look like
```python
MODEL:
    PRETRAINED: ./checkpoint.pth.tar # you need to specify checkpoint here
    OFA:
        CALIBRATE_BN: True
        NUM_SAMPLE_TRAINING: 1
        WIDTH_MULT: 0.25 # width multiplier
```

### ImageNet Experiments

#### Pre-training and Linear Evaluation
- Pre-training stage using US3L (c.f. [scripts/imagenet/run_byol_featdistill_imagenet.sh](scripts/imagenet/run_byol_featdistill_imagenet.sh)), run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/imagenet/us_resnet50_byol_featdistill_imagenet.yaml \
    --output /data/train_log_OFA/imagenet/r50/ours_base_mse_distill_mse_100ep -j 8
```

- Linear evaluation on ImageNet (c.f. [scripts/imagenet/run_linear_eval_imagenet.sh](scripts/imagenet/run_linear_eval_imagenet.sh)). If you want to specify the model weights and width multiplier, see the instructions on CIFAR above and modify the corresponding [configs/imagenet/resnet18_linear_eval_imagenet.yaml](configs/imagenet/resnet18_linear_eval_imagenet.yaml).


## Join Us
Welcome to be a member (or an intern) of our team if you are interested in Quantization, Pruning, Distillation, Self-Supervised Learning and Model Deployment.
Please send your resume to sunpeiqin@megvii.com.

## Citation
Please consider citing our work in your publications if it helps your research.
```
@article{US3L,
   title         = {Three Guidelines You Should Know for Universally Slimmable Self-Supervised Learning},
   author        = {Yun-Hao Cao, Peiqin Sun and Shuchang Zhou},
   year          = {2023},
   booktitle={The IEEE Conference on Computer Vision and Pattern Recognition},}
```