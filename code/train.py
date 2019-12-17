from lstm import lstm
from word2vec import word2vec_train
from dataset import loadfile,clean_data
import keras.utils
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import imp
import numpy as np
import yaml
import sys

imp.reload(sys)
np.random.seed()
#参数

voc_dim = 150 #word的向量维度
lstm_input = 150#lstm输入维度
epoch_time = 10#epoch
batch_size = 32 #batch

def data2inx(w2indx,X_Vec):
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)

        data.append(new_txt)
    return data
def train_lstm(model, x_train, y_train, x_test, y_test):
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',#hinge
                  optimizer='adam', metrics=['mae', 'acc'])

    print("Train..." )
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time, verbose=1)

    print("Evaluate...")
    print(model.predict(x_test))
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('../model/lstm_java.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save('../model/lstm_java_total.h5')
    print('Test score:', score)

print("开始清洗数据................")
clean_data('../data/angry.txt','../data/angry_clean.txt')
clean_data('../data/anxious.txt','../data/anxious_clean.txt')
clean_data('../data/depress.txt','../data/depress_clean.txt')
clean_data('../data/sad.txt','../data/sad_clean.txt')
print("清洗数据完成................")
print("开始下载数据................")
X_Vec, y=loadfile()
print("下载数据完成................")
print("开始构建词向量................")
input_dim,embedding_weights,w2dic = word2vec_train(X_Vec)
print("构建词向量完成................")

index = data2inx(w2dic,X_Vec)
index2 = sequence.pad_sequences(index, maxlen=voc_dim )
x_train, x_test, y_train, y_test = train_test_split(index2, y, test_size=0.2)
y_train = keras.utils.to_categorical(y_train, num_classes=4)
y_test = keras.utils.to_categorical(y_test, num_classes=4)
#
model=lstm(input_dim, embedding_weights)
train_lstm(model, x_train, y_train, x_test, y_test)
