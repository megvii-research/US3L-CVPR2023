python -m torch.distributed.launch --nproc_per_node=4 --use_env main_lincls_ddp.py \
  -a us_vit_tiny -b 256 \
  --dist-url 'env://' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained logs/mocov3_ours_vit_tiny_cifar10/checkpoint.pth.tar \
  --width-mult 0.3 --output logs/linear_eval
