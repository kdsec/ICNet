#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
@author: zhang zhenyu
@contact: zzysay@gmail.com
@file: segment.py
@time: 2018/12/11 17:27
@desc:
"""

import jieba

jieba.load_userdict('../data/rule/stop_list.txt')
cur_path = '../'


def chinese_segment():
    with open(cur_path + 'data/classify.txt', 'r', encoding='utf8') as fr, \
            open(cur_path + 'data/classify_data.seg', 'w', encoding='utf8') as fw:
        for idx, line in enumerate(fr):
            print(idx)
            if line == '\n':
                continue
            line = line.strip().split('\t')
            seg_list = list(jieba.cut(''.join(line[:-1])))
            fw.write(' '.join(seg_list) + '\n')


if __name__ == '__main__':
    chinese_segment()
