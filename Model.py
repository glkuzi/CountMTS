
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 19:42:45 2018

@author: User
"""

import tensorflow as tf
import numpy as np
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, GlobalMaxPooling1D, Flatten, MaxPooling1D
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
#import keras
import json
NUM_NETS = 1
DROPOUT = 0.5
RDROPOUT = 0.2
L2 = 1e-4
SAVEDIR = '/models/'


#import DataGenerator
from DataGenerator import *

def main():
    dg = DataGenerator(train)
    valid = DataGenerator(validation)
    #inp = keras.Input(shape = (FRAMESIZE // 2 + 1, ), name = 'input_layer')
    model = Sequential()
    for i in range(NUM_NETS):
        model.add(Bidirectional(LSTM(
                81, #kernel_regularizer = regularizers.l2(L2), 
                #recurrent_regularizer = regularizers.l2(L2), 
                #bias_regularizer = regularizers.l2(L2), 
                activation = 'tanh', return_sequences = True), input_shape = (1,81)))
        model.add(Bidirectional(LSTM(
                81, #kernel_regularizer = regularizers.l2(L2), 
                #recurrent_regularizer = regularizers.l2(L2), 
                #bias_regularizer = regularizers.l2(L2), 
                return_sequences = True)))
        model.add(Bidirectional(LSTM(
                10, kernel_regularizer = regularizers.l2(L2), 
                recurrent_regularizer = regularizers.l2(L2), 
                bias_regularizer = regularizers.l2(L2), 
                return_sequences = True)))
    #model.add(GlobalMaxPooling1D())
    #model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(5, activation = 'softmax'))
    print(model)
    check = ModelCheckpoint('Euclidean_norm_weights.{epoch:02d}-{val_loss:.2f}.hdf5', period = 50)
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = optimizers.Adam(), metrics = ['accuracy'])
    model.fit_generator(generator = dg, epochs = 2070, verbose = 2, callbacks = [check],
                        validation_data = valid, max_queue_size = 1)
    model.save('mod.hdf5')
    return 0


main()