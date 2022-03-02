# Created by RitsukiShuto on 2022/02/28.
# convert to MFCC
#
import pandas as pd
import glob
import librosa

# 0:dog, 1:rooster, 2:pig, 3:cow
list_MFCC = []
list_label = []

# wavファイルを取得
file_list = glob.glob('./ESC-50/audio/*-0.wav')

# MFCC
for file_name in file_list:
    y, sr = librosa.core.load(file_name, sr = None)
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20)

    ceps = mfcc.mean(axis = 1)

    list_MFCC.append(ceps)
    list_label.append(0)


# wavファイルを取得
file_list = glob.glob('./ESC-50/audio/*-1.wav')    # 41 == 掃除機

# MFCC
for file_name in file_list:
    y, sr = librosa.core.load(file_name, sr = None)
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20)

    ceps = mfcc.mean(axis = 1)

    list_MFCC.append(ceps)
    list_label.append(1)

# wavファイルを取得
file_list = glob.glob('./ESC-50/audio/*-2.wav')

# MFCC
for file_name in file_list:
    y, sr = librosa.core.load(file_name, sr = None)
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20)

    ceps = mfcc.mean(axis = 1)

    list_MFCC.append(ceps)
    list_label.append(2)

# wavファイルを取得
file_list = glob.glob('./ESC-50/audio/*-3.wav')

# MFCC
for file_name in file_list:
    y, sr = librosa.core.load(file_name, sr = None)
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20)

    ceps = mfcc.mean(axis = 1)

    list_MFCC.append(ceps)
    list_label.append(3)

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