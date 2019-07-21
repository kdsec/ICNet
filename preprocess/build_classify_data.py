import json

import torch
from sklearn.model_selection import train_test_split

curr_path = '../'


def load_dictionary():
    index2word = torch.load(curr_path + 'data/dict.bin')
    word2index = {v: k for k, v in index2word.items()}
    return index2word, word2index


def load_data_v2():
    i2w, w2i = load_dictionary()
    data_o = open(curr_path + 'data/classify_data.seg', 'r', encoding='utf-8').readlines()
    data_t = open(curr_path + 'data/classify.tag', 'r', encoding='utf8').readlines()
    data = []
    for line, tag in zip(data_o, data_t):
        line = line.rstrip('\n').split()
        line = list(map(lambda token: w2i.get(token, 1), line))
        tag = tag.strip().split()
        data.append([line, tag])
    labels = [int(l.rstrip('\n')) for l in open(curr_path + 'data/classify_label.txt').read()]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    json.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
              open(curr_path + 'data/classify.tag.json', 'w'))


if __name__ == '__main__':
    load_data_v2()
