
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 19:42:45 2018

@author: User
"""

import tensorflow as tf
import numpy as np
from keras.layers import LSTM, Bidirectional, Dense, Flatten
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
#import keras
import json
from DataGenerator import *

NUM_NETS = 1
DROPOUT = 0.5
RDROPOUT = 0.2
L2 = 1e-6
SAVEDIR = '/models/'


def main():
    dg = DataGenerator(train)
    valid = DataGenerator(validation)
    '''
    Создаем модель
    '''
    model = Sequential()
    model.add(Bidirectional(LSTM(
            81, kernel_regularizer=regularizers.l2(L2),
            recurrent_regularizer=regularizers.l2(L2),
            bias_regularizer=regularizers.l2(L2), dropout=DROPOUT,
            recurrent_dropout=RDROPOUT,
            activation='tanh', return_sequences=True), input_shape=(1, 81)))
    model.add(Bidirectional(LSTM(
            81, kernel_regularizer=regularizers.l2(L2),
            recurrent_regularizer=regularizers.l2(L2),
            bias_regularizer=regularizers.l2(L2), dropout=DROPOUT,
            recurrent_dropout=RDROPOUT,
            return_sequences=True)))
    model.add(Bidirectional(LSTM(
            81, kernel_regularizer=regularizers.l2(L2),
            recurrent_regularizer=regularizers.l2(L2),
            bias_regularizer=regularizers.l2(L2), dropout=DROPOUT,
            recurrent_dropout=RDROPOUT,
            return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(5, kernel_regularizer=regularizers.l2(L2),
                    bias_regularizer=regularizers.l2(L2), activation='softmax'))
    check = ModelCheckpoint('mod.{epoch:02d}-{val_loss:.2f}.hdf5', period=50)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.00001), metrics=['accuracy'])
    model.fit_generator(generator=dg, steps_per_epoch=499, epochs=500, verbose=2, callbacks=[check],
                        validation_data=valid, max_queue_size=1)
    model.save('mod.hdf5')
    return 0

if __name__ == '__main__':
    main()
