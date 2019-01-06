# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:48:31 2018

@author: User
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile
import webrtcvad
import wave
import random
import json
import keras

FS = 16000
FRAMESIZE = 400
OVERLAP = 0.6
train = 'train.json'
validation = 'validation.json'
test = 'test.json'


'''
Важные параметры и зависимости:
    Аудиофайлы: формат .wav, PCM, 16kHz, bit depth = 16
'''


def stft(sig, frameSize=FRAMESIZE, overlapFac=OVERLAP, window=np.hanning):
    hop = int(frameSize - np.floor(overlapFac * frameSize))
    w = np.sqrt(window(frameSize))
    out = np.array([np.fft.rfft(w*sig[i:i+frameSize])
                    for i in range(0, len(sig)-frameSize, hop)])
    out = np.abs(out)
    out -= np.mean(out)
    return out


def activityTime(wf, frameduration=10):
    """
    Функция для вычисления промежутков, во время которых спикер говорит.
    Входные данные:
        wf - list, массив байтов, прочитанный аудиофайл;
        frameduration - int, длина промежутка для поиска, 10, 20 или 30 мс.
    Выходные данные:
        time - list, промежутки активности спикера, в мс.
    """
    vad = webrtcvad.Vad()  # создаем объект класса Vad
    vad.set_mode(1)  # уровень "агрессивности" алгоритма, отвечает за отделение промежутков, от 0 до 3
    pcmdata = wf.readframes(wf.getnframes())
    frame_size_vad = int(FS * frameduration / 1000) * 2  # окно длиной в frameduration мс
    K = int(np.floor(len(pcmdata) / frame_size_vad))
    time = []
    flag0 = False
    counter = 0
    trig = True
    for i in range(K):
        flag1 = vad.is_speech(pcmdata[frame_size_vad * i:
            frame_size_vad * (i + 1)], FS)
        if flag1 or flag0:
            if trig:
                time.append(counter)
                trig = False
        else:
            if not trig:
                time.append(counter)
                trig = True
        flag0 = flag1
        counter += 10
    if not trig:
        time.append(counter)
    time_new = []
    for i in range(len(time) // 2):
        time_new.append([time[2 * i], time[2 * i + 1]])
    return time_new


def onlyVoice(sig, time):
    """
    Функция для выделения голоса на записи.
    Входные данные:
        sig - array_like, входной сигнал;
        time - array_like, промежутки активности спикера.
    Выходные данные:
        clear_sig - array_like, сигнал без промежутков тишины.
    """
    data = []
    for i in range(len(time)):
        data.append(sig[int(time[i][0] * FS / 1000):
            int(time[i][1] * FS / 1000)])
    clear_sig = []
    for x in data:
        for y in x:
            clear_sig.append(y)
    return np.array(clear_sig)


def mixGenerate(data_dict, N=5, duration=5, shuffle=True):
    """
    Функция для генерации одной смешанной записи и разметки к ней.
    Входные данные:
        data_dict - dict, словарь с записями спикеров:
            key - string, speaker_id
            value - list, список всех записей спикера
        N-1 - int, максимальное количество спикеров в mix-е
        duration - int, длительность сигнала в секундах
        shuffle - boolean, флаг для перемешивания словаря
    Выходные данные:
        SumSignal - array_like, смешанный сигнал
        json_pathes - массив с разметкой активности для всех спикеров,
        которые смешаны в mix-е
    """
    keys = list(data_dict.keys())  # получаем ключи словаря
    if shuffle:
        random.shuffle(keys)  # перемешиваем ключи для выбора случайных записей
    num = np.random.randint(1, N)  # разыгрываем количество спикеров
    json_pathes = []  # массив для json-файлов разметки
    sumSignal = None  # суммарный сигнал
    cutoff = FS * duration  # 5 sec
    l0 = 0
    for i in range(num):  # получаем num случайных записей в массив pathes
        if l0 <= cutoff:
            while l0 <= cutoff:  # ищем файлы, соответсвующие минимальному размеру
                j = np.random.randint(0, len(keys))
                k = np.random.randint(0, len(data_dict[keys[j]]))  # разыгрываем номер записи для i-го спикера
                path = data_dict[keys[j]][k]
                #del (data_dict[keys[j]])[k]
                #if len(data_dict[keys[j]]) == 0:  # удаляем спикера, если у него не осталось записей
                #    data_dict.pop(keys[j])
                #    keys = list(data_dict.keys())
                fs, sig0 = scipy.io.wavfile.read(path)
                l0 = len(sig0)
        if l0 > cutoff:
            if sumSignal is None:  # суммируем сигналы
                sumSignal = sig0
            else:
                sumSignal = sumSignal[:cutoff+1] + sig0[:cutoff+1]
            path_json = path[:path.rfind('.')] + '.json'
            with open(path_json) as f:  # заполняем разметочный файл
                json_pathes.append(json.load(f))
        l0 = 0
    sumSignal = sumSignal[:cutoff+1] // max(abs(sumSignal))
    return sumSignal, json_pathes


def notation(sig, json_arr, overlapFac=OVERLAP):
    '''
    Функция для составления разметки.
    Входные данные:
        sig - array_like, смешанный сигнал
        json_arr - массив с разметкой активности для всех спикеров,
        которые смешаны в mix-е
    Выходные данные:
        speakerNumbers - array_like, target
    '''
    # длина stft массива
    tMax = int((len(sig) - FRAMESIZE) //
               (FRAMESIZE - np.floor(FRAMESIZE * OVERLAP))) + 1
    activities = []
    for x in json_arr:
        activities.append(x['activity'])
    speakerNumbers = []
    for i in range(tMax):
        count = 0
        for act in activities:
            for x in act:
                if i * 10 < x[1] and i * 10 >= x[0]:
                    count += 1
        speakerNumbers.append(count)
    return speakerNumbers


def voiceMarking(data_dict):
    '''
    Функция для разметки аудиофайлов на этапе генерации датасета
    Входные данные:
        data_dict - string, путь к директории с файлами
    '''
    keys = list(data_dict.keys())
    for key in keys:
        for path in data_dict[key]:
            wf = wave.open(path, 'rb')
            time = activityTime(wf)
            buf_dict = {"activity": [list(x) for x in time], "speaker_id": key}
            with open(path[:path.rfind('.')] + '.json', 'w') as fl:
                json.dump(buf_dict, fl)
    return 0


class DataGenerator(keras.utils.Sequence):
    '''
    Генератор для обучения модели.
    '''
    def __init__(self, train_json, batch_size=32, n_classes=5, duration=5, shuffle=True):
        '''
        Параметры:
            train_json - string, путь к .json файлу разметки заданной выборки
            batch_size - int, batch size
            n_classes - int, количество классов
            duration - int, длительность сигнала
        '''
        self.train_json = train_json
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.duration = duration
        self.data_dict = json.load(open(self.train_json))
        self.matrix = []
        self.notation = []
        for i in range(self.batch_size):
            buf = mixGenerate(self.data_dict,
                              duration=self.duration,
                              shuffle=self.shuffle)
            self.matrix.append(stft(buf[0]))
            self.notation.append(notation(buf[0], buf[1]))
        self.on_epoch_end()

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        X = []
        X.append(self.matrix[index])
        Y = []
        Y.append(self.notation[index])
        return np.array(X), keras.utils.to_categorical(Y, num_classes=self.n_classes)

    def on_epoch_end(self):
        self.matrix = []
        self.notation = []
        for i in range(self.batch_size):
            buf = mixGenerate(self.data_dict,
                              duration=self.duration,
                              shuffle=self.shuffle)
            self.matrix.append(stft(buf[0]))
            self.notation.append(notation(buf[0], buf[1]))


def main():
    return 0


if __name__ == '__main__':
    main()
