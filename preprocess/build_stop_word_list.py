def load_stop_word_list():
    path = ['../data/rule/descr_para.txt', '../data/rule/parameter.txt', '../data/rule/weapon.txt']
    dic1 = open(path[0], encoding='utf8').readlines()
    dic1 = set([s.strip() for s in dic1])
    dic2 = open(path[1], encoding='utf8').readlines()
    dic2 = set([s.strip() for s in dic2])
    dic3 = open(path[2], encoding='utf8').readlines()
    dic3 = set([s.strip() for s in dic3])
    dic = list(dic1 | dic2 | dic3)
    dic.sort()
    with open('../data/rule/stop_list.txt', 'w', encoding='utf8') as fw:
        for d in dic:
            if d:
                fw.write(d + '\n')


if __name__ == '__main__':
    load_stop_word_list()
