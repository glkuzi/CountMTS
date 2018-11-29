# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:13:08 2018

@author: User
"""

import numpy as np
from keras.models import load_model
import json
from DataGenerator import *


def main():
    model_name = 'Euclidean_norm_endless_adam_drop.250-1.64.hdf5'
    sig, json_arr = mixGenerate(json.load(open(test)), size=1, n=500,
                                              shuffle=True)
    matrix = []
    matrix.append(stft(sig))
    matrix0 = np.array(matrix / np.abs(matrix))
    target = notation(sig, json_arr)
    model = load_model(model_name)
    res = []
    for i in range(matrix[0].shape[0]):
        t = []
        t.append(matrix0[:, i])
        res.append(model.predict_classes(np.array(t)))
    final_res = []
    for x in res:
        final_res.append(x[0])
    print(final_res)
    print(target)
    return 0


if __name__ == '__main__':
    main()
