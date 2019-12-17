import re
import jieba
import numpy as np

def clean_data(rpath,wpath):
    # coding=utf-8
    pchinese = re.compile('([\u4e00-\u9fa5]+)+?')
    f = open(rpath,encoding="UTF-8")
    fw = open(wpath, "w",encoding="UTF-8")
    for line in f.readlines():
        m = pchinese.findall(str(line))
        if m:
            str1 = ''.join(m)
            str2 = str(str1)
            fw.write(str2)
            fw.write("\n")
    f.close()
    fw.close()
#clean_data('../data/angry.txt','../data/angry_new.txt')
#clean_data('../data/anxious.txt','../data/anxious_new.txt')
#clean_data('../data/depress.txt','../data/depress_new.txt')
#clean_data('../data/sad.txt','../data/sad_new.txt')
def loadfile():
    angry = []
    sad = []
    anxious=[]
    depress=[]
    with open('../data/angry_clean.txt',encoding='UTF-8') as f:
        for line in f.readlines():
            angry.append(list(jieba.cut(line, cut_all=False, HMM=True))[:-1])
    with open('../data/sad_clean.txt',encoding='UTF-8') as f:
        for line in f.readlines():
            sad.append(list(jieba.cut(line, cut_all=False, HMM=True))[:-1])
        f.close()
    with open('../data/anxious_clean.txt',encoding='UTF-8') as f:
        for line in f.readlines():
            anxious.append(list(jieba.cut(line, cut_all=False, HMM=True))[:-1])
        f.close()
    with open('../data/depress_clean.txt',encoding='UTF-8') as f:
        for line in f.readlines():
            depress.append(list(jieba.cut(line, cut_all=False, HMM=True))[:-1])
        f.close()

        X_Vec = np.concatenate((angry, sad,anxious,depress))

    y = np.concatenate((np.zeros(len(angry), dtype=int),
                        np.ones(len(sad), dtype=int),
                        2*np.ones(len(anxious), dtype=int),
                        3*np.ones(len(depress), dtype=int)))

    return X_Vec, y