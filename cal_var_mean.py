# 参考URL
# 物体検出とボケ検出で一眼レフ風の料理写真候補抽出
# https://qiita.com/tttttt/items/93fed9acf14dab59de65

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


"""
## データの読み込み
"""
# Model / data parameters
num_classes = 2
input_shape = (512, 512, 3)

test_ds = image_dataset_from_directory(
    directory='test_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(512, 512),
    shuffle=False,
    validation_split=None
    )


"""
## 取り込んだ画像を表示させる
"""

class_names = test_ds.class_names

for images, labels in test_ds:
    print("読み込んだ画像の枚数: ",len(images))

    # 元の画像
    plt.figure(figsize=(10, 10))
    for i in range(1):
        ax = plt.subplot(1, 1, i + 1)
        #軸を表示しない
        plt.xticks(color = "None")
        plt.yticks(color = "None")
        plt.tick_params(bottom = False, left = False)

        #表示          
        plt.imshow(images[i].numpy().astype("uint8"))            
        plt.axis("off")

    # バッチサイズが32なので一度にsubplotできる数は32まで
    plt.figure(figsize=(10, 10))
    for i in range(1):
        ax = plt.subplot(1, 1, i + 1)
        plt.xticks(color = "None")
        plt.yticks(color = "None")
        plt.tick_params(bottom = False, left = False)

        image_bgr = cv2.cvtColor(images[i].numpy().astype("uint8"), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)  
        hsv_value = hsv[:,:,2].mean()

        gray = cv2.cvtColor(images[i].numpy().astype("uint8"), cv2.COLOR_RGB2GRAY)
        gray_rgb_value = gray.var()
        gray_rgb_mean = gray.mean()
        
        #表示
        plt.imshow(gray, cmap='gray')
        plt.text(0, -20, gray_rgb_mean, fontsize=10)
        plt.text(0, -10, gray_rgb_value, fontsize=10)
        plt.axis("off")

plt.show()
