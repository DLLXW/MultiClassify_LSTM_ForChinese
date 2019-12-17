import numpy as np
import multiprocessing
import gensim
from gensim.models.word2vec import Word2Vec
voc_dim = 150 #word的向量维度
min_out = 4  #单词出现频率数
window_size = 7 #
cpu_count = multiprocessing.cpu_count()

def word2vec_train(X_Vec):
    model_word = Word2Vec(size=voc_dim,
                     min_count=min_out,
                     window=window_size,
                     workers=cpu_count,
                     iter=100)
    model_word.build_vocab(X_Vec)
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.iter)
    model_word.save('../model/Word2Vec_java.pkl')

    print(len(model_word.wv.vocab.keys()))
    input_dim = len(model_word.wv.vocab.keys()) + 1 #频数小于阈值的词语统统放一起，编码为0
    embedding_weights = np.zeros((input_dim, voc_dim))
    w2dic={}
    for i in range(len(model_word.wv.vocab.keys())):
        embedding_weights[i+1, :] = model_word[list(model_word.wv.vocab.keys())[i]]
        w2dic[list(model_word.wv.vocab.keys())[i]]=i+1
    return input_dim,embedding_weights,w2dic

