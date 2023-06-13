import numpy as np


def find_most_num(_list):
    most_num = -1
    most_value = -1
    value_num_dict = {}
    for i in _list:
        if i not in value_num_dict.keys():
            value_num_dict[i] = 1
        else:
            value_num_dict[i] += 1
        if value_num_dict[i] > most_num:
            most_num = value_num_dict[i]
            most_value = i
    return most_num,most_value

def calculate_label_acc(pseudo_label,gt_label):
    num_cluster = len(set(pseudo_label)) - (1 if -1 in pseudo_label else 0)
    acc = 0
    total = pseudo_label.shape[0]
    for label in range(num_cluster):
        index_label = np.where(pseudo_label == label)
        gt_value = gt_label[index_label]
        most_num, most_value = find_most_num(gt_value)
        acc += most_num

    return acc * 1.0 / total

