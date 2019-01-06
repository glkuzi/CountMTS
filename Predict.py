# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:13:08 2018

@author: User
"""

import numpy as np
from keras.models import load_model
import json
from DataGenerator import *


def accuracy_test(model_name, test=test):
    dg_test = DataGenerator(test)
    Num = 10
    model = load_model(model_name)
    acc = []
    for i in range(Num):
        length = dg_test.__len__()
        for j in range(length):
            X, Y = dg_test.__getitem__(j)
            Y_pred = model.predict_classes(X)
            Y_not = []
            for y in Y[0]:
                Y_not.append(list(y).index(1.))
            Y_diff = Y_pred[0] - np.array(Y_not)
            count = 0
            for y in Y_diff:
                if y == 0:
                    count += 1
            acc.append(count / len(Y_diff))
        dg_test.on_epoch_end()
    return np.mean(acc)


def predict_single_sample(model_name, test=test):
    sig, json_arr = mixGenerate(json.load(open(test)), duration=5,
                                              shuffle=True)
    matrix = stft(sig)
    target = notation(sig, json_arr)
    model = load_model(model_name)
    res = model.predict_classes(np.array([matrix]))
    print('Predicted notation:', res)
    print('Real notation:', target)
    diff = res[0] - np.array(target)
    count = 0
    for y in diff:
        if y == 0:
            count += 1
    acc = count / len(diff)
    print('Accuracy:', acc)


def main():
    model_name = '1_500_81_1_500_5_with_regs_0_6overlapPool_e-5_decay5e-4_abs_norm_peak_norm.600-1.36.hdf5'
    #print(accuracy_test(model_name='1_500_81_1_500_5_with_regs_0_6overlapPool_e-5_decay5e-4_abs_norm_peak_norm.600-1.36.hdf5'))
    predict_single_sample(model_name)
    return 0


if __name__ == '__main__':
    main()
