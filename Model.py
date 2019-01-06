
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 19:42:45 2018

@author: User
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import LSTM, Bidirectional, Dense, MaxPooling1D
from keras.models import Sequential, load_model
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from DataGenerator import *

NUM_NETS = 1
DROPOUT = 0.5
RDROPOUT = 0.2
L2 = 1e-6
SAVEDIR = '/models/'


def main():
    dg = DataGenerator(train)
    X, Y = dg.__getitem__(0)
    inp_shape = (X.shape[-2], X.shape[-1])
    print(inp_shape)
    valid = DataGenerator(validation)
    '''
    Создаем модель
    '''
    model = Sequential()
    model.add(Bidirectional(LSTM(
            81,
            activation='sigmoid', return_sequences=True),
            input_shape=inp_shape))
    model.add(Bidirectional(LSTM(
            81, return_sequences=True)))
    model.add(Bidirectional(LSTM(
            40, return_sequences=True)))
    model.add(MaxPooling1D(data_format='channels_first'))
    model.add(Dense(5, kernel_regularizer=regularizers.l2(L2),
                    bias_regularizer=regularizers.l2(L2),
                    activation='softmax'))
    check = ModelCheckpoint('1_498_201_1_498_5_with_regs_0_6overlapPool_e-5_decay5e-4_abs_norm_peak_norm_sig0mean.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.00001, decay=0.0005), metrics=['accuracy'])
    
    #model = load_model('1_500_81_1_500_5_with_regs_0_6overlapPool_e-5.500-1.41.hdf5')
    #model.compile(loss='categorical_crossentropy',
    #              optimizer=optimizers.Adam(lr=0.00001), metrics=['accuracy'])
    #check = ModelCheckpoint('1_500_81_1_500_5_with_regs_0_6overlapPool_e-5_decay_500+.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
    history = model.fit_generator(generator=dg, steps_per_epoch=32, epochs=50, verbose=2, callbacks=[check],
                        validation_data=valid, max_queue_size=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('1_498_201_1_498_5_with_regs_0_6overlapPool_e-5_decay5e-4_abs_norm_peak_norm.csv', index=False, index_label=False)
    return 0

if __name__ == '__main__':
    main()
