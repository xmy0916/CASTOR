# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import copy
import os.path
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from collections import OrderedDict


import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features_mask,extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import remap_list
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.cluster import calculate_label_acc
from clustercontrast.utils.watch_nvidia import send_msg

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None,mask=None,labels=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer,labels=labels,height=height,width=width,args=args),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,mask=None,labels=None):
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
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer,mask=mask, height=height,width=width,labels=labels),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type, feature_before_bn=args.use_bnn)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model

def create_camera_model():
    model = models.create("resnet_ibn50a", num_features=0, norm=True, dropout=0,
                          num_classes=0, pooling_type="gem")
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    main_worker(args)

def ratio_decay(epoch, total_epoch=120, init=0.2, min=0.1):
    p = float(epoch) / total_epoch
    scale = init - min
    ratio = init - (2. / (1. + np.exp(-10 * p)) - 1) * scale
    return ratio

def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    if args.resume:
        model_dict = torch.load(args.resume)
        print("==> resume from {},mAP is {}".format(args.resume,model_dict["best_mAP"]))
        model.load_state_dict(model_dict["state_dict"])

    # Evaluator
    evaluator = Evaluator(model, args)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model, use_teacher=args.use_teacher, use_bnn=args.use_bnn)

    # save np
    save_ones = False
    gt_labels = []
    cid_label = []
    for f, pid, cid in sorted(dataset.train):
       gt_labels.append(pid)
       cid_label.append(cid)
    gt_labels = np.array(gt_labels)


    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            # prepare DBscan
            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # extract features & get pseudo label for student model
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            # calculate camera dist matrix
            if args.use_camera:
                print("==> prepare camera dist matrix...")
                if not os.path.exists(args.camera_model_dir):
                    os.makedirs(args.camera_model_dir)
                npy_dir = os.path.join(args.camera_model_dir,"camera_dist.npy")
                if not os.path.isfile(npy_dir):
                    print("==> Camera dist matrix not found, start calculating...")
                    model_camera = create_camera_model()
                    static_dict = torch.load(os.path.join(args.camera_model_dir,"model_best.pth.tar"))["state_dict"]
                    model_camera.load_state_dict(static_dict, strict=False)
                    model_camera.eval()
                    camera_feature, _ = extract_features(model_camera, cluster_loader, print_freq=50)
                    camera_feature = torch.cat([camera_feature[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                    camera_dist = compute_jaccard_distance(camera_feature, k1=30, k2=6)
                    np.save(npy_dir, camera_dist)
                    del camera_feature, static_dict, camera_dist, model_camera
                camera_dist = np.load(npy_dir).astype(np.float32)
                ratio = ratio_decay(epoch,init=args.cam_init,min=args.cam_min)
                print("Camera parameter is {}".format(ratio))
                rerank_dist = rerank_dist - ratio * camera_dist
                del camera_dist

            rerank_dist = np.where(rerank_dist < 0, 0.0, rerank_dist) # 防止小于0
            pseudo_labels = cluster.fit_predict(rerank_dist)

            # pull back uncluster points
            if args.use_pb:
                print("==> Start pulling back uncluster points...")
                total, wrong = 0, 0
                for turn in range(args.pb_turns):
                    for anchor_index in np.where(pseudo_labels == -1)[0]:
                        anchor_dist = rerank_dist[anchor_index]
                        #topk_index = np.argpartition(anchor_dist, topk + 1)
                        topk_index = (-anchor_dist).argsort()[::-1][:args.pb_topk + 1]
                        for index, i in enumerate(pseudo_labels[topk_index]):
                            if i != -1:
                                if anchor_index in (-rerank_dist[topk_index[index]]).argsort()[::-1][: args.pb_topk + 1] \
                                        and anchor_dist[topk_index[index]] <= 0.9999:
                                    total += 1
                                    pseudo_labels[anchor_index] = pseudo_labels[topk_index[index]]
                                    if gt_labels[topk_index[index]] != gt_labels[anchor_index]:
                                        wrong += 1
                                break
                    print("Round {} the correct pull back number is {}|{}".format(turn + 1, total - wrong, total))
            print("==> Finish pulling uncluster points back!")
            acc = calculate_label_acc(pseudo_labels, gt_labels)
            print("==> epoch:{}' label acc is {}".format(epoch, acc))
            del rerank_dist

            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        pseudo_labeled_dataset = []
        for i, ((fname, gt_label, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid, gt_label))

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers_labels = sorted(centers.keys())
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)
            return centers, centers_labels

        cluster_features, cluster_labels = generate_cluster_features(pseudo_labels, features)
        del cluster_loader, features

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, loss_func=args.loss,temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard, use_teacher=args.use_teacher).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        memory.labels = torch.Tensor(cluster_labels).long().cuda()
        trainer.memory = memory
        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        # sleep for dataloader
        time.sleep(0.5)
        train_loader.new_epoch()
        time.sleep(0.5)

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
            try:
                send_msg(args.email,"{} epoch:{} mAP:{} best_mAP:{}".format(args.msg,epoch,mAP,best_mAP))
            except:
                pass

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.000175,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--iters', type=int, default=800)
    parser.add_argument('--step-size', type=int, default=20)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='avg')
    parser.add_argument('--use-hard', action="store_true")

    # my params
    parser.add_argument('--use-camera', action="store_true")
    parser.add_argument('--msg', type=str, default="164_baseline")
    parser.add_argument('--loss', type=str, default="ce")
    parser.add_argument('--opt', type=str, default="Adam")
    parser.add_argument('--use-teacher', action="store_true")
    parser.add_argument('--use-bnn', action="store_true")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--cam-eval', type=float, default=0.1)
    parser.add_argument('--cam-min', type=float, default=0.0)
    parser.add_argument('--cam-init', type=float, default=0.1)
    parser.add_argument('--use-pb', action="store_true")
    parser.add_argument('--pb-topk', type=int, default=7)
    parser.add_argument('--pb-turns', type=int, default=5)
    parser.add_argument('--camera-model-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--email', type=str, default=None)





    main()
