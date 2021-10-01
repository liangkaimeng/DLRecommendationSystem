# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: LiangKaimeng
# Date  : 2021/9/28

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, layers, utils
from sklearn.model_selection import train_test_split








data = pd.read_csv("data/bike_data.csv", encoding="gbk")

data['Model'] = data['Model'].astype(str).apply(lambda x: x.replace("Women's Mountain Shorts", '未知名称'))
data['Model'] = data['Model'].astype(str).apply(lambda x: x.replace("(", ""))
data['Model'] = data['Model'].astype(str).apply(lambda x: x.replace(")", ""))
data['Model'] = data['Model'].astype(str).apply(lambda x: x + " ")

df = data.groupby(['OrderNumber'])['Model'].sum().reset_index()
df['Model'] = df['Model'].astype(str).apply(lambda x: x.strip())

df['count'] = df['Model'].astype(str).apply(lambda x: x.split(' ')).apply(len)

texts = df[df['count']>=2].Model.tolist()

all_texts = texts[0]
for i in texts[1:]:
    all_texts += " " + i

all_texts = list(set(all_texts.split(' ')))
all_texts.append("<PAD>")
all_texts = set(all_texts)

text_to_id = {c: i for i, c in enumerate(all_texts)}
id_to_text = {i: c for i, c in enumerate(all_texts)}

df['Model'] = df['Model'].astype(str).apply(lambda x: "<PAD> <PAD> <PAD> " + x)

df['Model'] = df['Model'].astype(str).apply(lambda x: x.split(' '))

x = []
y = []

for values in df['Model'].tolist():
    if len(values) == 5:
        x.append(" ".join(values[:4]))
        y.append(values[-1])
    else:
        for i in np.arange(0, len(values)-4, step=1):
            x.append(" ".join(values[i: i+4]))
            y.append("".join(values[i+4]))

texts = []
for text in x:
    texts.append([text_to_id[i] for i in text.split(' ')])

label = [text_to_id[i] for i in y]

texts = np.array(texts)
label = np.array(label)

texts = texts / len(all_texts)

x = texts.reshape(texts.shape[0], 4, 1)
y = tf.keras.utils.to_categorical(label)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=2021)

model = Sequential([
    layers.LSTM(units=800, input_shape=(4, 1), return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(units=800, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(units=800),
    layers.Dropout(0.2),
    layers.Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y,
          # validation_data=(x_test, y_test),
          epochs=500,
          batch_size=64)


