from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.evaluators import Evaluator
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.evaluators import Evaluator, extract_features_mask,extract_features
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from collections import OrderedDict

def get_data(name, data_dir, height, width, batch_size, workers, mask=None, labels=None, args=None):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer,norm_flag=False,mask=mask,labels=labels,height=256,width=128,args=args),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

def create_camera_model():
    model = models.create("resnet_ibn50a", num_features=0, norm=True, dropout=0,
                          num_classes=0, pooling_type="gem")
    # use CUDA
    model.cuda()
    static_dict = torch.load("logs/market1501/camera/model_best.pth.tar")["state_dict"]
    del static_dict["module.classifier.weight"]

    new_state_dict = OrderedDict()
    for k, v in static_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    for param in model.parameters():
        param.detach_()
    return model


def main_worker(args):
    cudnn.benchmark = True
    global model_camera

    log_dir = osp.dirname(args.resume)
    # sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset, test_loader = get_data(args.dataset, args.data_dir, args.height,
                                    args.width, args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)

    # camera_mask_dict_all, labels_all = extract_features_mask(model_camera, test_loader, print_freq=50) 
    # dataset, test_loader = get_data(args.dataset, args.data_dir, args.height,
                                    # args.width, args.batch_size, args.workers,mask=camera_mask_dict_all, labels=labels_all, args=args)

    if args.dsbn:
        print("==> Load the model with domain-specific BNs")
        convert_dsbn(model)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')

    if args.dsbn:
        print("==> Test with {}-domain BNs".format("source" if args.test_source else "target"))
        convert_bn(model, use_target=(not args.test_source))

    model.cuda()
    model = nn.DataParallel(model)

    # Evaluator
    model.eval()
    evaluator = Evaluator(model,args)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, rerank=args.rerank)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('--resume', type=str,
                        default="/media/yixuan/DATA/cluster-contrast/market-res50/logs/model_best.pth.tar",
                        metavar='PATH')
    # testing configs
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--dsbn', action='store_true',
                        help="test on the model with domain-specific BN")
    parser.add_argument('--test-source', action='store_true',
                        help="test on the source domain")
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/media/yixuan/Project/guangyuan/workpalces/SpCL/examples/data')
    parser.add_argument('--camera-model-dir', type=str, metavar='PATH',
                        default='/media/yixuan/Project/guangyuan/workpalces/SpCL/examples/data')
    parser.add_argument('--cam-eval', type=float, default=0.1)
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-camera', action="store_true")
    parser.add_argument('--use-pose', action="store_true")
    parser.add_argument('--embedding_features_path', type=str,
                        default='/media/yixuan/Project/guangyuan/workpalces/SpCL/embedding_features/mark1501_res50_ibn/')
    main()
