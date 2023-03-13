python -m torch.distributed.launch --nproc_per_node=8 --use_env main_moco_ofa_ddp.py \
  -a us_vit_tiny -b 512 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'env://' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --moco-mlp-dim 2048 \
  --min-width 0.25 --max-width 1.0 \
  --output logs/mocov3_ours_vit_tiny_cifar10
