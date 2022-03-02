# Created by RitsukiShuto on 2022/03/01.
# data argmentation
#
import pandas as pd
import glob
import librosa

# 0:dog, 1:rooster, 2:pig, 3:cow
list_MFCC = []
list_label = []

for i in range (0, 4):
    file_list = glob.glob('./ESC-50/audio/*' + '-' + str(i) + '.wav')

    # convert to MFCC
    for file_name in file_list:
        # nomal
        y, sr = librosa.core.load(file_name, sr = None)
        mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20)

        ceps = mfcc.mean(axis = 1)

        list_MFCC.append(ceps)
        list_label.append(i)



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