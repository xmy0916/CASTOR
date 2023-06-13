export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=1,0 YOURPATH/reid/bin/python -u examples/cluster_contrast_train_usl.py \
  --use-camera \
  --cam-init 0.05  \
  --cam-min 0.0  \
  --use-pb  \
  --pb-topk 7 \
  --pb-turns 5  \
  --iters 800 \
  -d msmt17 \
  --eval-step 10 \
  -j 2 \
  --eps 0.7 \
  -b 128 \
  --logs-dir ./logs/msmt17/ \
  --camera-model-dir ./examples/pretrained/camera_model/msmt17/ \
  --msg msmt17
