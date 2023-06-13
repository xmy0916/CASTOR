from __future__ import print_function, absolute_import

import copy
import json
import os.path as osp
import shutil
import numpy as np

import torch
from torch.nn import Parameter

from .osutils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar',is_camera=False):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        if is_camera:
            shutil.copy(fpath, osp.join(osp.dirname(fpath), 'camera_model_best.pth.tar'))
        else:
            shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model

def remove_pseudo_labels(pseudo_labels, ema_pseudo_labels, alpha=0.2):
    print("==>start removing uncertain point, alpha is {}:".format(alpha))
    remove_num = 0
    result = []
    for index, label in enumerate(pseudo_labels):
        if label == -1:
            result.append(label)
            continue
        index_pseudo_labels = np.where(pseudo_labels == label)[0]
        index_ema_pseudo_labels = np.where(ema_pseudo_labels == ema_pseudo_labels[index])[0]

        iou_s = len((set(index_pseudo_labels) & set(index_ema_pseudo_labels))) * 1.0 \
              / (len(set(index_pseudo_labels))+0.0001)

        # iou_ema = len((set(index_pseudo_labels) & set(index_ema_pseudo_labels))) * 1.0 \
        #      / (len(set(index_ema_pseudo_labels))+0.0001)
        # iou_ema = 0.0
        # iou = iou_s + iou_ema
        if iou_s < alpha:
            result.append(-1)
            remove_num += 1
        else:
            result.append(label)
    print("finished removed uncertain points:{}".format(remove_num))
    return remap_list(np.array(result))

def remap_list(target_list):
    tmp_list = copy.deepcopy(target_list)
    tmp_dict = {}
    for index, item in enumerate(target_list):
        if item == -1:
            continue
        if item not in tmp_dict.keys():
            tmp_dict[item] = [index]
        else:
            tmp_dict[item].append(index)

    for index, k in enumerate(sorted(tmp_dict.keys())):
        tmp_list = np.where(target_list == k, index, tmp_list)

    return tmp_list
