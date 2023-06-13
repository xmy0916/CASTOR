from __future__ import absolute_import

import copy
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import torch
from clustercontrast.utils.data import transforms as T
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None,mask=None,labels=None,height=256,width=128,args=None,norm_flag=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mask = mask
        self.height = height
        self.width = width
        # self.resize = T.Resize((height, width), interpolation=3)
        # self.model_cam = model_cam
        self.mean = (0.4914, 0.4822, 0.4465)
        self.args=args
        self.labels = labels
        self.norm_flag = norm_flag
        self.normal_trans = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        if len(self.dataset[index]) == 4:
            fname, pid, camid, gt_label = self.dataset[index]
        else:
            fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        if self.norm_flag:
            wo_trans_img = copy.deepcopy(img)
            wo_trans_img = self.normal_trans(wo_trans_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.norm_flag:
            if len(self.dataset[index]) == 4:
                return (img,wo_trans_img), fname, pid, camid, index, gt_label
            else:
                return (img,wo_trans_img), fname, pid, camid, index
        else:
            if len(self.dataset[index]) == 4:
                return img, fname, pid, camid, index, gt_label
            else:
                return img, fname, pid, camid, index

    def transform_convert(self, img_tensor, transform,name):
        """
        param img_tensor: tensor
        param transforms: torchvision.transforms
        """
        if 'Normalize' in str(transform):
            normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
            mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
            std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
            img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

        img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

        if 'ToTensor' in str(transform) or img_tensor.max() < 1:
            img_tensor = img_tensor.detach().numpy() * 255

        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.numpy()

        if img_tensor.shape[2] == 3:
            img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
            img.save(name)
        elif img_tensor.shape[2] == 1:
            img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
        else:
            raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

