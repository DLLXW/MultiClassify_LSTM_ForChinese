import keras.utils
from keras import utils as np_utils
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation

lstm_input = 150#lstm输入维度
voc_dim = 150 #word的向量维度
def lstm(input_dim, embedding_weights):
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,
                        input_dim=input_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=lstm_input))
    model.add(LSTM(256, activation='softsign'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model
