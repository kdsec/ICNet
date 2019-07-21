#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
@author: zhang zhenyu
@contact: zzysay@gmail.com
@file: indicator_tagging.py
@time: 2018/12/11 16:51
@desc:
"""
import re
cur_path = '../data/'


def build_dic():
    path = ['rule/descr_para.txt', 'rule/parameter.txt', 'rule/weapon.txt']
    dic1 = open(cur_path+path[0], encoding='utf8').readlines()
    dic1 = set([s.strip() for s in dic1])
    dic2 = open(cur_path+path[1], encoding='utf8').readlines()
    dic2 = set([s.strip() for s in dic2])
    dic3 = open(cur_path+path[2], encoding='utf8').readlines()
    dic3 = set([s.strip() for s in dic3])
    return dic1, dic2, dic3


if __name__ == '__main__':
    dic1, dic2, dic3 = build_dic()
    with open(cur_path + 'classify_data.seg', 'r', encoding='utf8') as fr, \
            open(cur_path + 'classify.tag', 'w', encoding='utf8') as fw:
        for line in fr:
            line = line.strip().split()
            tag_list = list()
            for word in line:
                if word in dic1:
                    _type = '1'
                elif word in dic2:
                    _type = '2'
                elif word in dic3:
                    _type = '3'
                elif re.match('[0-9]+(\.)?[0-9]*', word):
                    _type = '4'
                else:
                    _type = '0'
                tag_list.append(_type)
            fw.write(' '.join(tag_list) + '\n')

