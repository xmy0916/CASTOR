from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):

        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ErasingBackground(object):
    def __init__(self, model_camera, probability=0.5, avg_mask_num=10, threshold=0.5, mean=(0.4914, 0.4822, 0.4465)):
        self.model_camera = model_camera
        self.probability = probability
        self.mean = mean
        self.avg_mask_num = avg_mask_num
        self.threshold = threshold

    def __call__(self, img):
        # mask [2048 16 8]
        # img [3 256 128]
        if random.uniform(0, 1) >= self.probability:
            return img

        with torch.no_grad():
            w,h = img.shape[1],img.shape[2]
            # feature_mask = mask.cpu().numpy() # 2048 16 8
            img_c = img.unsqueeze(0).cuda() # 1 3 256 128
            feature_mask = self.model_camera(img_c).cpu().numpy() # 1 2048 16 8
            # print("erase!!!")
            choice_pos = np.random.randint(0,feature_mask.shape[1],[self.avg_mask_num])
            choice_masks = feature_mask[0][choice_pos]
            avg_mask = np.mean(choice_masks,axis=0)
            mask = cv2.resize(avg_mask, (h, w))
            normed_mask = mask / mask.max()
            index = normed_mask > self.threshold
            img[0,index] = self.mean[0]
            img[1,index] = self.mean[1]
            img[2,index] = self.mean[2]
            # img = torch.from_numpy(img)
            # exit()

        return img