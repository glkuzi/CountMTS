# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:48:31 2018

@author: User
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile
import webrtcvad, wave, random, json
import keras

FS = 16000
FRAMESIZE = 160
train = 'train.json'
validation = 'validation.json'


'''
Важные параметры и зависимости:
    Аудиофайлы: формат .wav, PCM, 16kHz, bit depth = 16
'''


def stft(sig, frameSize=FRAMESIZE, overlapFac=0, window=np.hanning):
    hop = int(frameSize - np.floor(overlapFac * frameSize))
    w = np.sqrt(window(frameSize))
    out = np.array([np.fft.rfft(w*sig[i:i+frameSize])
                    for i in range(0, len(sig)-frameSize, hop)])
    return out


def stft1(X):
    window = 'hann'
    noverlap = 0 # FRAMESIZE // 2
    Y = scipy.signal.stft(X, FS, window=window, nperseg=FRAMESIZE,
                          noverlap=noverlap)
    print(len(Y[1]))
    return Y[2]


def activityTime(wf, frameduration = 10):
    """
    Функция для вычисления промежутков, во время которых спикер говорит.
    Входные данные:
        wf - list, массив байтов, прочитанный аудиофайл;
        frameduration - int, длина промежутка для поиска, 10, 20 или 30 мс.
    Выходные данные:
        time - list, промежутки активности спикера, в мс.
    """
    vad = webrtcvad.Vad() # создаем объект класса Vad
    vad.set_mode(1) # уровень "агрессивности" алгоритма, отвечает за отделение промежутков, от 0 до 3
    pcmdata = wf.readframes(wf.getnframes())
    frame_size_vad = int(FS * frameduration / 1000) * 2 # окно длиной в frameduration мс
    K = int(np.floor(len(pcmdata) / frame_size_vad))
    time = []
    flag0 = False
    counter = 0
    trig = True
    for i in range(K):
        flag1 = vad.is_speech(pcmdata[frame_size_vad * i :
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


def mixGenerate(data_dict, N=5, shuffle=True):
    """
    Функция для генерации одной смешанной записи и разметки к ней.
    Входные данные:
        data_dict - dict, словарь с записями спикеров:
            key - string, speaker_id
            value - list, список всех записей спикера
        N - int, максимальное количество спикеров в mix-е
    Выходные данные:
        SumSignal - array_like, смешанный сигнал
        json_pathes - массив с разметкой активности для всех спикеров,
        которые смешаны в mix-е
    """
    keys = list(data_dict.keys()) # получаем ключи словаря
    if shuffle:
        random.shuffle(keys) # перемешиваем ключи для выбора случайных записей
    num = np.random.randint(1, N) # разыгрываем количество спикеров
    print(num)
    json_pathes = [] # массив для json-файлов разметки
    sumSignal = None # суммарный сигнал
    for i in range(num): # получаем num случайных записей в массив pathes
        l = 0
        while l < 320:
            k = np.random.randint(0, len(data_dict[keys[i]])) # разыгрываем номер записи для i-го спикера
            path = data_dict[keys[i]][k] 
            if len(data_dict[keys[i]]) == 0: # удаляем спикера, если у него не осталось записей
                data_dict[keys[i]].pop(keys[i])
            fs, sig = scipy.io.wavfile.read(path) # считываем сигнал
            if sumSignal is None: # суммируем сигналы
                l = len(sig)
                if l >= 320:
                    sumSignal = sig
            else:
                l = min(len(sig), len(sumSignal))
                #if l < 160:
                #    break
                #sumSignal = [sum(x) for x in zip(sumSignal[:l], sig[:l])]
                if l >= 320:
                    sumSignal = sumSignal[:l] + sig[:l]
        path_json = path[:path.rfind('.')] + '.json'
        with open(path_json) as f: # заполняем разметочный файл
            json_pathes.append(json.load(f))
    sumSignal = sumSignal // num
    return sumSignal, json_pathes


def notation(sig, json_arr):
    tMax = len(sig) * 100 // FS # количество отрезков в 10 мс
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
    def __init__(self, train_json, batch_size=2, n_classes=5, shuffle=True):
        self.train_json = train_json
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.sig, self.json_arr = mixGenerate(json.load(open(self.train_json)))
        self.matrix = stft(self.sig)
        self.notation = notation(self.sig, self.json_arr)
        #print(self.matrix)
        #print(self.notation)
        self.on_epoch_end()

    def __getitem__(self, index):
        X = []
        Y = []
        X.append(self.matrix[index * self.batch_size:self.batch_size * (index + 1)])
        Y.append(self.notation[index * self.batch_size:self.batch_size * (index + 1)])
        #X = self.matrix[index * self.batch_size:self.batch_size * (index + 1)]
        #Y = self.notation[index * self.batch_size:self.batch_size * (index + 1)]
        return np.array(X / np.abs(X)), keras.utils.to_categorical(Y, num_classes = self.n_classes)#np.array(Y)#keras.utils.to_categorical(Y, num_classes = self.n_classes)
    
    def __len__(self):
        return int(np.floor((np.shape(self.matrix)[0] / self.batch_size)))
    
    def on_epoch_end(self):
        self.sig, self.json_arr = mixGenerate(json.load(open(self.train_json)),
                                          shuffle = self.shuffle)
        self.matrix = stft(self.sig)
        self.notation = notation(self.sig, self.json_arr)


def main():
    #train_file = 'train.txt'
    #data = json.load(open(train))
    #sig, js = mixGenerate(data)
    #print(js)
    #scipy.io.wavfile.write('sum.wav', FS, sig)
    #print(np.shape(stft(sig)))
    #print(len(sig))

    #print(notation(sig, js))
    #print(len(notation(sig, js)))
    dg = DataGenerator(train)
    X, Y = dg.__getitem__(0)
    print(X.shape, Y.shape)
    #print(X, Y)
    #testfile = 'ru_0022.wav'
    #fs, sig = scipy.io.wavfile.read(testfile)
    #print(np.max(sig))
    #vad = webrtcvad.Vad()
    #vad.set_mode(1)
    #wf = wave.open(testfile, 'rb')
    #time = activityTime(wf)
    #scipy.io.wavfile.write('voices.wav', fs, onlyVoice(sig, time))
    
    #dim = (32, 32, 32)
    #print(np.empty((32, *dim)))
    return 0


main()