export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=1,0 YOURPATH/reid/bin/python -u examples/cluster_contrast_train_usl.py \
  -b 128 \
  --cam-eval 0.1 \
  --cam-init 0.1 \
  --cam-min 0.0 \
  --use-camera \
  --camera-model-dir ./examples/pretrained/camera_model/market1501/ \
  --use-hard \
  --use-pb \
  --pb-topk 2 \
  --pb-turns 5 \
  --lr 0.000175 \
  --iters 800 \
  -d market1501 \
  -j 2 \
  --eps 0.4 \
  --logs-dir ./logs/market1501/ \
  --msg market1501
