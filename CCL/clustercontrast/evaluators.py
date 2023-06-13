from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
import torch.nn as nn
from clustercontrast.models import create

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch
from clustercontrast.models.dsbn import CBN2d,CBN1d
import os


def create_camera_model():
    model = create("resnet_ibn50a", num_features=0, norm=True, dropout=0,
                          num_classes=0, pooling_type="gem")
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model

def extract_cnn_feature_mask(model, inputs):
    inputs = to_torch(inputs).cuda()
    probs,outputs = model(inputs)
    outputs = outputs.data.cpu()
    probs = probs.data.cpu()
    return probs,outputs

def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs

def extract_cnn_feature_dl(model, inputs):
    inputs = to_torch(inputs).cuda()
    final_fea,b3_fea = model(inputs)
    final_fea, b3_fea = final_fea.data.cpu(),b3_fea.data.cpu()
    return final_fea, b3_fea

def extract_features_dl(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    final_feas = OrderedDict()
    b3_feas = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            final_fea, b3_fea = extract_cnn_feature_dl(model, imgs)
            for fname, output_final,output_b3, pid in zip(fnames, final_fea, b3_fea, pids):
                final_feas[fname] = output_final
                b3_feas[fname] = output_b3
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return final_feas, b3_feas, labels

def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels

def extract_features_mask(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            probs,outputs = extract_cnn_feature_mask(model, imgs)
            for fname, output, pid, prob in zip(fnames, outputs, pids, probs):
                features[fname] = output
                labels[fname] = torch.argmax(prob)


            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def extract_features_camera(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cid, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            CBN2d.camera_index = cid
            CBN1d.camera_index = cid
            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features,query=None, gallery=None):
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

def pairwise_distance_camera(features,camera_features,query=None, gallery=None):
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    x_camera = torch.cat([camera_features[f].unsqueeze(0) for f, _, _ in query], 0)
    x += x_camera
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    y_camera = torch.cat([camera_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    y += y_camera
    del x_camera,y_camera
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model, args=None):
        super(Evaluator, self).__init__()
        self.model = model
        self.use_camera = args.use_camera
        self.args = args

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)

        if self.use_camera:
            print("==> prepare camera dist matrix for eval...")
            if not os.path.exists(self.args.camera_model_dir):
                os.makedirs(self.args.camera_model_dir)
            npy_dir = os.path.join(self.args.camera_model_dir, "q_g_dist_mat.npy")
            if not os.path.isfile(npy_dir):
                print("==> Camera dist matrix for eval not found, start calculating...")
                model_camera = create_camera_model()
                static_dict = torch.load(os.path.join(self.args.camera_model_dir, "model_best.pth.tar"))["state_dict"]
                model_camera.load_state_dict(static_dict, strict=False)
                model_camera.eval()
                camera_feature, _ = extract_features(model_camera, data_loader, print_freq=50)
                distmat_camera, _, _ = pairwise_distance(camera_feature, query, gallery)
                np.save(npy_dir, distmat_camera)
                del camera_feature, static_dict, distmat_camera, model_camera
            distmat_camera = np.load(npy_dir)
            distmat -= self.args.cam_eval * distmat_camera
            del distmat_camera

        distmat = np.where(distmat < 0, 0.0, distmat) # 防止小于0
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)

        distmat_qq -= 0.1 * distmat_qq_camera
        distmat_gg -= 0.1 * distmat_gg_camera
        
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)


class CameraEvaluator(object):
    def __init__(self, model, camera_model,args=None):
        super(CameraEvaluator, self).__init__()
        self.model = model
        self.camera_model = camera_model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        features_camera, _ = extract_features(self.camera_model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance_camera(features,features_camera, query, gallery)

        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
                               cmc_flag=cmc_flag)
        return results


class Evaluator_Camera(object):
    def __init__(self, model, use_camera):
        super(Evaluator_Camera, self).__init__()
        self.model = model
        self.use_camera = use_camera

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features_camera(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)

        if self.use_camera:
            distmat_camera = np.load("q_g_dist_mat.npy")
            distmat -= 0.1 * distmat_camera
            del distmat_camera

        distmat = np.where(distmat < 0, 0.0, distmat)  # 防止小于0
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
