import cv2
import time
import argparse
import os
import torch
import numpy as np

import os.path as osp
from clustercontrast import datasets

from clustercontrast.evaluators import extract_features
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast import models
from torch.utils.data import DataLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from torch import nn


def create_camera_model():
    model = models.create("resnet_ibn50a", num_features=0, norm=True, dropout=0,
                          num_classes=0, pooling_type="gem")
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


parser = argparse.ArgumentParser()
# path
working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--data-dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'data'))
parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                    choices=datasets.names())
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
args = parser.parse_args()


def main():
    # model = posenet.load_model(args.model)
    # model = model.cuda()
    model_camera = create_camera_model()
    static_dict = torch.load("logs/market1501/camera/model_best.pth.tar")["state_dict"]
    del static_dict["module.classifier.weight"]
    model_camera.load_state_dict(static_dict)

    # output_stride = model.output_stride
    dataset = get_data(args.dataset, args.data_dir)

    # 计算训练时相机矩阵
    cluster_loader = get_test_loader(dataset, 256, 128, 64, 4, testset=sorted(dataset.train))
    camera_feature, _ = extract_features(model_camera, cluster_loader, print_freq=50)
    camera_feature = torch.cat([camera_feature[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    camera_dist = compute_jaccard_distance(camera_feature, k1=30, k2=6)
    np.save("camera_dist.npy", camera_dist)
    del camera_feature, _, camera_dist




    # 计算评估时相机矩阵
    # test_loader = get_test_loader(dataset, 256, 128, 64, 4)
    # features_camera, _ = extract_features(model_camera, test_loader)
    # distmat_camera, _, _ = pairwise_distance(features_camera, dataset.query, dataset.gallery)
    #
    # np.save("q_g_dist_mat.npy", distmat_camera)
    # del distmat_camera, features_camera, test_loader
    #
    # print("finish")



if __name__ == "__main__":
    main()
