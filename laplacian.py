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
import glob


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

    for i in range(len(images)):
        # 元の画像
        ax = plt.subplot(1, 2, i + 1)
        #軸を表示しない
        plt.xticks(color = "None")
        plt.yticks(color = "None")
        plt.tick_params(bottom = False, left = False)

        #表示          
        plt.imshow(images[i].numpy().astype("uint8"))     
        plt.axis("off")

        # グレースケール
        ax = plt.subplot(1, 2, i + 2)
        plt.xticks(color = "None")
        plt.yticks(color = "None")
        plt.tick_params(bottom = False, left = False)

        image_bgr = cv2.cvtColor(images[i].numpy().astype("uint8"), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)  
        hsv_value = hsv[:,:,2].mean()

        gray = cv2.cvtColor(images[i].numpy().astype("uint8"), cv2.COLOR_RGB2GRAY)

        gamma = 1.25       
        gamma_cvt = np.zeros((256,1),dtype = 'uint8')
        for i in range(256):
            gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
        
        gray = cv2.LUT(gray,gamma_cvt)

        image_bgr = (gray - np.mean(image_bgr))/np.std(image_bgr)*16+64
        image_bgr = image_bgr.astype(np.uint8)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # gray = gray.astype(np.uint8)

        gray_canny = cv2.Canny(gray, 20, 150)
        print(cv2.countNonZero(gray))
        print(cv2.countNonZero(gray_canny))
        number = cv2.countNonZero(gray_canny) / cv2.countNonZero(gray)
        
        # temp_img = cv2.resize(temp_img, dsize=(420, 420))
        
        gray_laplacian_1 = cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var() 
        gray_laplacian_3 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var() 
        gray_laplacian_5 = cv2.Laplacian(gray, cv2.CV_32F, ksize=5).var() 
        
        # 重心のx方向から20%くらいの幅の画素数を見て、前後方向か横方向か判断する
        # 前後方向の場合、画面下側の見切れた判断の範囲を大きくする
        # 画素の総数で割って正規化が必要？
        # 重心位置を求める
        mu = cv2.moments(gray, False)
        x,y= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])

        # 画像をトリミング
        # trimming_image = gray[y-100:y+100, x-100:x+100]

        # 図形の描画
        # cv2.circle(gray, (x,y), 10, color=(255, 0, 0), thickness=3)
        # cv2.rectangle(gray,(x-30, y-30), (x+30, y+30), (255, 255, 255), thickness=2, lineType  =cv2.LINE_8, shift=0)

        plt.imshow(gray_canny, cmap="gray")
        # plt.imshow(gray, cmap="gray")
        # plt.text(0, -20, gray_laplacian, fontsize=10)
        plt.text(0, -70, "x = {}".format(x), fontsize=10)
        plt.text(0, -130, "y = {}".format(y), fontsize=10)
        plt.axis("off")

        # ax = plt.subplot(1, 3, i + 3)
        # plt.imshow(trimming_image, cmap="jet")
        # plt.axis("off")

        print(number)
        print(gray_laplacian_1)
        print(gray_laplacian_3)
        print(gray_laplacian_5)

plt.show()
