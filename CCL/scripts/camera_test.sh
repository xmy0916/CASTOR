export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=1 python3 -u examples/test_camera.py --alpha 0.1 -b 32 -a resnet50 -d market1501 --epochs 120 --momentum 0.1 --eps 0.4 --num-instances 16 --pooling-type avg --use-hard --logs-dir ./logs/market1501/camera_r50
