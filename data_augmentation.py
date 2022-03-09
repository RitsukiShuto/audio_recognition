# Created by RitsukiShuto on 2022/03/01.
# data argmentation
#
from lib2to3.pytree import convert
from audiomentations import Compose, AddGaussianNoise, Shift
import librosa as lr

import os
import random
import glob

import pandas as pd
import numpy as np

def convert_MFCC(file_name, i):
    # convert to MFCC
    print('Converting to' + file_name)
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20)
    ceps = mfcc.mean(axis = 1)

    list_MFCC.append(ceps)
    list_label.append(i)

def WhiteNoise(file_name, sr):
    lr.audio.sf.write(file_name, file_name, sr)
    convert_MFCC(file_name, i)

def seed_set(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# 0:dog, 1:rooster, 2:pig, 3:cow
list_MFCC = []
list_label = []

seed_set()

for i in range (0, 4):
    file_list = glob.glob('./ESC-50/audio/*' + '-' + str(i) + '.wav')
    for file_name in file_list:
        y, sr = lr.core.load(file_name, sr = None)
        convert_MFCC(file_name, i)  # Nomal
        # WhiteNoise(file_name, i)           # WhiteNoise


# データフレーム化
# 20次元のMFCCデータフレームを作成
df_ceps = pd.DataFrame(list_MFCC)

columuns_name = []  # カラム名を"dict + 番号"で示す
for i in range(20):
    columuns_name_tmp = 'dict{0}'.format(i)
    columuns_name.append(columuns_name_tmp)

df_ceps.columns = columuns_name

# ラベルのデータフレームを作成
df_label = pd.DataFrame(list_label, columns=['label'])

# ヨコ後方にconcat
df = pd.concat([df_label, df_ceps], axis = 1)
df.to_csv("data/raw/MFCC.csv")