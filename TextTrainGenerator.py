# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:22:44 2018

@author: User
"""

import glob
import json
from DataGenerator import *


train_file = 'train.txt'
val_file = 'validation.txt'
test_file = 'test.txt'
FILENAMES = ['train.txt', 'validation.txt', 'test.txt']
VOLUME_TRAIN = 0.7  # объемы выборок
VOLUME_VALID = 0.2
VOLUME_TEST = 0.1


def main():
    '''
    Для каждой выборки создается текстовый файл, в котором содержатся пути
    к файлам и speaker_id. Также формируется .json файл с аналогичным названием,
    содержащий данные по выборке в формате словаря, key - speaker_id,
    value - список с путями к записям спикера. На основе этого .json файла
    для каждой записи создается разметка с моментами активности спикера при
    помощи функции voiceMarking. Датасет:
    http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/
    '''
    source_dir = './SpeechCountDataset'
    source_dirs0 = glob.glob(source_dir + '/**')
    cutoff_train = int(VOLUME_TRAIN * len(source_dirs0))
    cutoff_test = int(VOLUME_TEST * len(source_dirs0))
    source_dirs = []
    source_dirs.append(source_dirs0[:cutoff_train])
    source_dirs.append(source_dirs0[cutoff_train:len(source_dirs0) - cutoff_test])
    source_dirs.append(source_dirs0[len(source_dirs0) - cutoff_test:])
    for i in range(len(FILENAMES)):
        with open(FILENAMES[i], 'w') as fl:
            for dirs in source_dirs[i]:
                f = open(dirs + '/etc/README')
                for line in f:
                    if line.find('User Name:') != -1:
                        spk_id = line[line.find(':') + 1:line.find('Speaker')]
                    else:
                        break
                files = glob.glob(dirs + '/wav/*.wav')
                for file in files:
                    print(file + ' ' + spk_id, file=fl)
        tf = open(FILENAMES[i])
        data = {}  # список с элементами вида ['path', 'speaker_id']
        for line in tf:
            val, key = line.split()
            if key not in data:
                data[key] = []
            data[key].append(val)
        with open(FILENAMES[i][:-4] + '.json', 'w') as fl:
            json.dump(data, fl)
        voiceMarking(data)
        tf.close()
    return 0


if __name__ == '__main__':
    main()
