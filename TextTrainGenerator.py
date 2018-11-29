# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:22:44 2018

@author: User
"""

import glob, json
from DataGenerator import *


train_file = 'train.txt'
val_file = 'validation.txt'
test_file = 'test.txt'
FILENAMES = ['train.txt', 'validation.txt', 'test.txt']
VOLUME_TRAIN = 0.7
VOLUME_VALID = 0.2
VOLUME_TEST = 0.1

def main():
    source_dir = './SpeechCountDataset'
    source_dirs0 = glob.glob(source_dir + '/**')
    cutoff_train = int(VOLUME_TRAIN * len(source_dirs0))
    cutoff_valid = int(VOLUME_VALID * len(source_dirs0))
    cutoff_test = int(VOLUME_TEST * len(source_dirs0))
    source_dirs_valid = source_dirs0[cutoff_train:len(source_dirs0) - cutoff_test]
    #source_dirs = source_dirs[:cutoff_train]
    source_dirs_train = source_dirs0[len(source_dirs0) - cutoff_test:]
    #print(source_dirs)
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
                        #print(spk_id)
                    else:
                        break
                files = glob.glob(dirs + '/wav/*.wav')
                for file in files:
                    print(file + ' ' + spk_id, file = fl)
        tf = open(FILENAMES[i])
        data = {} # список с элементами вида ['path', 'speaker_id']
        for line in tf:
            #print(line)
            val, key = line.split()
            if key not in data:
                data[key] = []
            data[key].append(val)
        with open(FILENAMES[i][:-4] + '.json', 'w') as fl:
            json.dump(data, fl)
        #print(data)
        voiceMarking(data)
        tf.close()
    '''
    with open(val_file, 'w') as fl:
        for dirs in source_dirs_valid:
            f = open(dirs + '/etc/README')
            for line in f:
                if line.find('User Name:') != -1:
                    spk_id = line[line.find(':') + 1:line.find('Speaker')]
                    #print(spk_id)
                else:
                    break
            files = glob.glob(dirs + '/wav/*.wav')
            for file in files:
                print(file + ' ' + spk_id, file = fl)
    
    
    tf = open(train_file)
    data = {} # список с элементами вида ['path', 'speaker_id']
    for line in tf:
        #print(line)
        val, key = line.split()
        if key not in data:
            data[key] = []
        data[key].append(val)
    with open(train_file[:-4] + '.json', 'w') as fl:
        json.dump(data, fl)
    #print(data)
    voiceMarking(data)
    tf.close()
    
    vf = open(val_file)
    valid_data = {} # список с элементами вида ['path', 'speaker_id']
    for line in vf:
        #print(line)
        val, key = line.split()
        if key not in valid_data:
            valid_data[key] = []
        valid_data[key].append(val)
    with open(val_file[:-4] + '.json', 'w') as fl:
        json.dump(valid_data, fl)
    #print(data)
    voiceMarking(valid_data)
    vf.close()
    '''
    return 0

if __name__ == '__main__':
    main()