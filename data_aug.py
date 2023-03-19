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
from keras.preprocessing.image import ImageDataGenerator
import scipy

"""
## 機能の切り替えフラグ
## データ拡張した画像を保存する: saveNewFigure
## 画像を切り取る: cropImage
##
"""
# ###################################################################################
saveNewFigure = True
cropImage = True
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

bg_img = cv2.imread('./test_data/*.jpeg')

"""
## ImageDataGeneratorの設定
## 使用するdatageneratorのコメントアウトを外して使用すること
"""
# # 回転させる
# datagen = ImageDataGenerator(rotation_range = 60)
# # ランダムに上下反転する。
# datagen = image.ImageDataGenerator(vertical_flip=False) 
# # ランダムに左右反転する。
# datagen = image.ImageDataGenerator(horizontal_flip=True)
# [-0.3 * Height, 0.3 * Height] の範囲でランダムに上下平行移動する。
datagen = image.ImageDataGenerator(height_shift_range=0.5)
# # [-0.3 * Width, 0.3 * Width] の範囲でランダムに左右平行移動する。
# datagen = image.ImageDataGenerator(width_shift_range=0.6)
# # -5° ~ 5° の範囲でランダムにせん断する。 
# datagen = image.ImageDataGenerator(shear_range=5)
# # [1 - 0.3, 1 + 0.3] の範囲でランダムに拡大縮小する。
# datagen = image.ImageDataGenerator(zoom_range=0.3)
# # [-5.0, 5.0] の範囲でランダムに画素値に値を足す。
# datagen = image.ImageDataGenerator(channel_shift_range=5.)
# # [0.3, 1.0] の範囲でランダムに明度を変更する。
# datagen = ImageDataGenerator(brightness_range=[0.3, 0.4])
# datagen = ImageDataGenerator(channel_shift_range = 100)

# datagen = ImageDataGenerator(rotation_range = 40, vertical_flip=True,horizontal_flip=True,height_shift_range=0.3, width_shift_range=0.3,channel_shift_range = 100)


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

        if cropImage:
            offset_height = 0
            offset_width = 512
            target_height = 1024
            target_width = 512
            cropped_img = tf.image.crop_to_bounding_box(images[i], offset_height, offset_width, target_height, target_width)

            bg_img = cv2.resize(cropped_img.numpy(),(1024,1024))

            # if isinstance(cropped_img, type(None)):
            # bg_img[0:1024, 0:512] = cropped_img


            # dx, dy = 512, 0
            # M = np.float32([[1,0,dx],[0,1,dy]])

            # cropped_img = cv2.warpAffine(np.array(images[i]), M, dsize=(1024, 1024), borderMode=cv2.BORDER_TRANSPARENT, dst=bg_img.copy())

            # img=cropped_img[np.newaxis, :, :, :]
            img=bg_img[np.newaxis, :, :, :]
        else:
            # (Height, Width, Channels)  -> (1, Height, Width, Channels) 
            img=images[i][np.newaxis, :, :, :]

        # 画像を切り取る際は以降の処理でImageDataGeneratorを無効にするために、ImageDataGeneratorを意味のないものにする
        if cropImage:
            datagen = image.ImageDataGenerator(vertical_flip=False)

        # ImageDataGeneratorを適用、1枚しかないので、ミニバッチ数は1
        if saveNewFigure:
            gen = datagen.flow(img, batch_size=1, save_to_dir=save_path, save_prefix='generated', save_format='png')
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
