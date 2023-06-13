export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 YOURPATH/reid/bin/python -u examples/test.py \
  -b 64 \
  --use-camera \
  --cam-eval 0.12 \
  -a resnet50 \
  -d msmt17 \
  --pooling-type avg  \
  --data-dir ./examples/data \
  --resume ../best_model/CCL/msmt17.pth.tar \
  --camera-model-dir ./examples/pretrained/camera_model/msmt17/

