![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.8](https://img.shields.io/badge/PyTorch->=1.8-blue.svg)

# Pseudo Label Rectification with Joint Camera Shift Adaptation and Outlier Progressive Recycling for Unsupervised Person Re-Identification [[pdf]](https://ieeexplore.ieee.org/document/9967431)
The *official* repository for [Pseudo Label Rectification With Joint Camera Shift Adaptation and Outlier Progressive Recycling for Unsupervised Person Re-Identification](https://ieeexplore.ieee.org/document/9967431) TITS'22.

## Requirements

### Installation
We provide packaged python packages, which can be directly downloaded and unzipped to your server.

address: [python3](https://aistudio.baidu.com/aistudio/datasetdetail/185465https://aistudio.baidu.com/aistudio/datasetdetail/185465)
```shell
# you can use this python package to run python scripts like:
/yourpath/reid/bin/python hello_world.py
# pip install something like:
/yourpath/reid/bin/python -m pip install numpy
```
> [INFO] This environment has packaged all the required dependencies. Please do not modify it at will！

### Prepare Datasets

Please make sure your dataset path is as follows：
```text
## CCL
/YourPath/CASTOR/CCL/examples
├── data
│ ├── dukemtmcreid
│ │ └── DukeMTMC-reID
│ ├── market1501
│ │ └── Market-1501-v15.09.15
│ └── msmt17
│     └── MSMT17_V1
```

Download the datasets:

For privacy reasons, we don't have the copyright of the dataset. Please contact authors to get this dataset.

```
DukeMTMC-reID/
├── bounding_box_test
├── bounding_box_train
└── query

Market-1501-v15.09.15/
├── bounding_box_test
├── bounding_box_train
├── gt_bbox
├── gt_query
└── query

MSMT17_V1/
├── list_gallery.txt  
├── list_query.txt  
├── list_train.txt  
├── list_val.txt 
├── train
└── test

```

## Pre-trained Models
- When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and save it under the path of `CASTOR/CCL/examples/pretrained/` and `CASTOR/IDM/examples/pretrained/`.
- ImageNet-pretrained models for ResNet-50 will be automatically downloaded in the python script.
- You need download our trained camera classification model from this [link](https://aistudio.baidu.com/aistudio/datasetdetail/185472) and unzip it to `CASTOR/CCL/examples/pretrained/` and `CASTOR/IDM/examples/pretrained/`

Please make sure your pretrain models path is as follows：

```text
## CCL
/YourPath/CASTOR/CCL/examples/pretrained
├── resnet50_ibn_a.pth.tar (if you want to train IBN-ResNet)
├── resnet50-19c8e357.pth (it will be automatically downloaded by the python script)
├── camera_model/
│ ├── market1501
│ │ └── model_best.pth.tar
│ ├── dukemtmc
│ │ └── model_best.pth.tar
│ ├── msmt17
│ │ └── model_best.pth.tar
```

## ReID performance

### Unsupervised ReID (CCL baseline)

##### Market-1501
```shell
cd CASTOR/CCL/
# train command
sh scripts/market1501.sh
# test command refer to https://github.com/alibaba/cluster-contrast-reid
# View script to determine model save path!!!
sh scripts/test_market1501.sh
```
|    Method    | mAP  | Rank-1 | Rank-5 | Rank-10 |                               Download                                |
|:------------:|:----:|:------:|:------:|:-------:|:---------------------------------------------------------------------:|
|     CCL      | 82.1 |  92.3  |  96.7  |  97.9   |       [link](https://github.com/alibaba/cluster-contrast-reid)        |
| CCL + CASTOR | 86.2 |  94.8  |  98.4  |  98.8   | [model+log](https://aistudio.baidu.com/aistudio/datasetdetail/202712) |

##### DUKEMTMC-reID
```shell
cd CASTOR/CCL/
# train command
sh scripts/dukemtmc.sh
# test command refer to https://github.com/alibaba/cluster-contrast-reid
# View script to determine model save path!!!
sh scripts/test_dukemtmc.sh
```
|    Method    | mAP  | Rank-1 | Rank-5 | Rank-10 |                         Download                         |
|:------------:|:----:|:------:|:------:|:-------:|:--------------------------------------------------------:|
|     CCL      | 72.6 |  84.9  |  91.9  |  93.9   | [link](https://github.com/alibaba/cluster-contrast-reid) |
| CCL + CASTOR | 75.5 |  88.6  |  93.7  |  95.0   |                           [model+log](https://aistudio.baidu.com/aistudio/datasetdetail/202713)                           |

##### MSMT17
```shell
cd CASTOR/CCL/
# train command
sh scripts/msmt17.sh
# test command refer to https://github.com/alibaba/cluster-contrast-reid
# View script to determine model save path!!!
sh scripts/test_msmt17.sh
```
|    Method    | mAP  | Rank-1 | Rank-5 | Rank-10 |                         Download                         |
|:------------:|:----:|:------:|:------:|:-------:|:--------------------------------------------------------:|
|     CCL      | 27.6 |  56.0  |  66.8  |  71.5   | [link](https://github.com/alibaba/cluster-contrast-reid) |
| CCL + CASTOR | 37.3 |  70.4  |  80.1  |  83.5   |                      [model+log](https://aistudio.baidu.com/aistudio/datasetdetail/224038)                       |


## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [CCL](https://github.com/alibaba/cluster-contrast-reid)
- [IDM](https://github.com/SikaStar/IDM)
- [SPCL](https://github.com/yxgeee/SpCL)
- [ICE](https://github.com/chenhao2345/ICE)
- [IICS](https://github.com/SY-Xuan/IICS)


## Citation

If you find this code useful for your research, please cite our paper

```
@ARTICLE{9967431,
  author={Xu, Mingyuan and Guo, Haiyun and Jia, Yuheng and Dai, Zhitao and Wang, Jinqiao},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Pseudo Label Rectification With Joint Camera Shift Adaptation and Outlier Progressive Recycling for Unsupervised Person Re-Identification}, 
  year={2022},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TITS.2022.3224233}}
```

## Contact

If you have any question, please feel free to contact us. E-mail: [xmy0916@bupt.edu.cn](xmy0916@bupt.edu.cn).