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
    model_name = '1_500_81_1_500_5_with_regs_0_6overlapPool_e-5_decay5e-4_abs_norm_peak_norm.600-1.36.hdf5'#'wide_samples_250_long_without_regs.13100-1.26.hdf5'
    size = 500
    sig, json_arr = mixGenerate(json.load(open(test)), duration=5,
                                              shuffle=True)
    matrix = stft(sig)
    print(0)
    matrix0 = matrix#np.array(np.log(matrix / np.abs(matrix) + 1e-5))
    print(matrix0.shape)
    target = notation(sig, json_arr)
    model = load_model(model_name)
    res = []
    for i in range(1, matrix.shape[0] // size):
        t = []
        t.append(matrix0[(i - 1) * size:i * size, :])
        res.append(model.predict_classes(np.array(t)))
    matr = []
    matr.append(matrix0)
    res = model.predict_classes(np.array(matr))
    '''
    final_res = []
    for x in res:
        final_res = final_res + list(x[0])
    print(final_res)
    '''
    print(res)
    print(target)
    return 0


if __name__ == '__main__':
    main()
