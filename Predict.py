# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:13:08 2018

@author: User
"""

#import tensorflow as tf
import numpy as np
#from keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, GlobalMaxPooling1D, Flatten, MaxPooling1D
from keras.models import Sequential, load_model
#from keras import regularizers
#from keras import optimizers
#import keras
import json
NUM_NETS = 1
DROPOUT = 0.5
RDROPOUT = 0.2
L2 = 1e-4
from DataGenerator import *


def main():
    #test_gen = DataGenerator(test)
    model_name = 'weights.450-0.66.hdf5'
    sig, json_arr, data_dict = mixGenerate(json.load(open(test)), 
                                              size = 1,
                                              n = 500,
                                              shuffle = True)
    print(1)
    matrix = []
    matrix.append(stft(sig) / np.linalg.norm(stft(sig)))
    #print(matrix)
    matrix0 = np.array(matrix) #/ np.abs(matrix))
    target = notation(sig, json_arr)
    print(2)
    model = load_model(model_name)
    print(3)
    res = []
    print(matrix0.shape)
    for i in range(matrix[0].shape[0]):
        t = []
        t.append(matrix0[:, i])
        res.append(model.predict_classes(np.array(t)))
    final_res = []
    for x in res:
        final_res.append(x[0])
    print(final_res)
    #print(res)
    print(target)
    return 0


if __name__ == '__main__':
    main()