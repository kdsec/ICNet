#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
@author: zhang zhenyu
@contact: zzysay@gmail.com
@file: pre_classify_data.py
@time: 2018/12/11 18:02
@desc:
"""
import os

import numpy as np

curr_path = '../data/'


def load_data():
    # Split by words
    pos_path = curr_path + 'positive_final.txt'
    neg_path = curr_path + 'negtive_final.txt'
    print(os.getcwd())
    positive_examples = list(open(pos_path, 'r', encoding='utf8').readlines())
    negative_examples = list(open(neg_path, 'r', encoding='utf8').readlines())
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    return [positive_examples, positive_labels, negative_examples, negative_labels]


def build_input_data(x, y):
    x = np.array(x)
    y = np.array(y)
    np.random.seed(55)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # x_train = x_shuffled[:int(len(y)*0.8)]
    # x_test = x_shuffled[int(len(y)*0.8):]
    # y_train = y_shuffled[:int(len(y)*0.8)]
    # y_test = y_shuffled[int(len(y)*0.8):]

    with open(curr_path + 'classify.txt', 'w', encoding='utf8') as fw:
        for x, y in zip(x_shuffled.tolist(), y_shuffled.tolist()):
            fw.write(x.strip() + '\t' + str(y))
            fw.write('\n')

    with open(curr_path + 'classify_data.txt', 'w', encoding='utf8') as fw:
        for x in x_shuffled.tolist():
            fw.write(x.strip())
            fw.write('\n')

    with open(curr_path + 'classify_label.txt', 'w', encoding='utf8') as fw:
        for y in y_shuffled.tolist():
            fw.write(str(y))


if __name__ == '__main__':
    p_x, p_y, n_x, n_y = load_data()
    build_input_data(p_x + n_x, p_y + n_y)
