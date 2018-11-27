# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:22:44 2018

@author: User
"""

import glob, json
from DataGenerator import *


train_file = 'train.txt'
val_file = 'validation.txt'
VOLUME = 0.8


def main():
    source_dir = './SpeechCountDataset'
    source_dirs = glob.glob(source_dir + '/**')
    cutoff = int(VOLUME * len(source_dirs))
    source_dirs_valid = source_dirs[cutoff:]
    source_dirs = source_dirs[:cutoff]
    #print(source_dirs)
    with open(train_file, 'w') as fl:
        for dirs in source_dirs:
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
    return 0

if __name__ == '__main__':
    main()