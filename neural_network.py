# Created by RitsukiShuto on 2022/03/01.
# neural network
#
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import utils as np_utils

#CSVファイルの読み込み
data_set = pd.read_csv("./data/raw/MFCC.csv", sep=",", header=0)
data_set = data_set.values

x = data_set[:, 2:]      # 説明関数を抽出
y = data_set[:, 1]       # 目的関数を抽出

print(x)
print(y)

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)
y_train = np_utils.to_categorical(y_train, 4)
y_test = np_utils.to_categorical(y_test, 4)

# ニューラルネットワークを定義
model = Sequential()
model.add(InputLayer(input_shape=(20,)))   # 入力層
model.add(Dense(16, activation='softmax')) # 中間層
model.add(Dense(10, activation='softmax'))
model.add(Dense(4, activation='softmax'))  # 出力層
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 学習
epochs = 150
batch_size = 16
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# テスト
score = model.evaluate(x_test, y_test)
print()
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])