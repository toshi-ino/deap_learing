# JFIF以外のファイルだとコードが動きません
# data_aug.pyでデータ拡張を行う前にJFIF以外のファイルを削除すること

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
import scipy

"""
## 機能の切り替えフラグ
## データ拡張した画像を保存する: saveNewFigure
## 画像を切り取る: cropImage
##
"""
# ###################################################################################
saveNewFigure = True
cropImage = False
# ###################################################################################


"""
## データの読み込み
"""
test_ds = image_dataset_from_directory(
    directory='test_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(1024, 1024),
    shuffle=False,
    validation_split=None
    )

# images = cv2.imread('./test_data/*.jpeg')

"""
## ガンマ補正
"""
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


"""
## 作成した画像の保存先の設定
"""
# 指定したディレクトリが存在しないとエラーになるので、
# 予め作成しておく。
save_path = './test_data/output'
import os
os.makedirs(save_path, exist_ok=True)


"""
## 画像の生成と生成した画像の保存を行う
"""
class_names = test_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in test_ds:

    rangeNumber = len(images) if saveNewFigure else 9
    for i in range(rangeNumber):

        img=images[i][np.newaxis, :, :, :]
        # ImageDataGeneratorを適用、1枚しかないので、ミニバッチ数は1
        if saveNewFigure:
            gamma = 0.85  # 要調整
            img_corrected = adjust_gamma(images[i].numpy(), gamma)
            img=img_corrected[np.newaxis, :, :, :]

            datagen = image.ImageDataGenerator()
            gen = datagen.flow(img, batch_size=1, save_to_dir=save_path, save_prefix='generated', save_format='jpeg')
            batches = next(gen)
        else:
            gen = datagen.flow(img.numpy(), batch_size=1)
            # next関数を使う、なぜ使うかはわかっていない
            batches = next(gen) 
            # 画像として表示するため、3次元データにし、float から uint8 にキャストする。
            gen_img = batches[0].astype(np.uint8)

            ax = plt.subplot(3, 3, i + 1)
            # 軸を表示しない
            plt.xticks(color = "None")
            plt.yticks(color = "None")
            plt.tick_params(bottom = False, left = False)
            # 表示
            plt.imshow(gen_img)
            
            plt.title(class_names[tf.math.argmax(labels[i]).numpy()])
            plt.axis("off")

if not saveNewFigure:
    plt.show()
