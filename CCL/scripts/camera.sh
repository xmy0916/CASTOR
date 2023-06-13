export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=1 python3 -u examples/train_camera.py --alpha 0.1 -b 64 -a resnet_ibn50a -d msmt17 --epochs 120 --momentum 0.1 --eps 0.4 --num-instances 16 --pooling-type gem --logs-dir ./logs/market1501/camera_msmt17
