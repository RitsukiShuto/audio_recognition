# Created by RitsukiShuto on 2022/03/01.
# train.py
#
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('./data/raw/MFCC.csv')
arr = df.values

X = arr[:, 2:]      # 説明関数を抽出
y = arr[:, 1]       # 目的関数を抽出

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print (model.score(X_test, y_test))