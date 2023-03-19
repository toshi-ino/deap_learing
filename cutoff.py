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
# num_classes = 2
# input_shape = (512, 512, 3)

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

    for i in range(1):

        files = glob.glob("./test_data/test/*")
        for file in files:
            images = cv2.imread(file)

        # 元の画像
        ax = plt.subplot(1, 3, i + 1)
        #軸を表示しない
        plt.xticks(color = "None")
        plt.yticks(color = "None")
        plt.tick_params(bottom = False, left = False)

        #表示          
        plt.imshow(images)     
        plt.axis("off")

        # グレースケール
        ax = plt.subplot(1, 3, i + 2)
        plt.xticks(color = "None")
        plt.yticks(color = "None")
        plt.tick_params(bottom = False, left = False)

        # image_bgr = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        # hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)  
        # hsv_value = hsv[:,:,2].mean()

        gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
        whitePixels_gray = cv2.countNonZero(gray)
        
        # 重心のx方向から20%くらいの幅の画素数を見て、前後方向か横方向か判断する
        # 重心位置を求める
        mu = cv2.moments(gray, False)
        x_m,y_m= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])

        # 色がついたピクセルを確認する範囲
        direction_judgement_band = images.shape[1] * 0.2
        p1 = x_m - direction_judgement_band
        p2 = x_m + direction_judgement_band

        # 画像をトリミング
        trimming_image = gray[0:images.shape[0], int(p1):int(p2)]
        whitePixels = cv2.countNonZero(trimming_image)

        # 色がついたピクセルの比率
        ratio_whitePixels = round(whitePixels / whitePixels_gray, 2)

        # 簡易な前後判断
        front_back_flg = True if ratio_whitePixels >= 0.8 else False

        print(gray.shape[0],gray.shape[1])

        # 外接矩形の演算 
        x,y,w,h = cv2.boundingRect(gray)

        print("変更前　x,y,w,h =",x,y,w,h)


        # 画像の端の画素をカウント
        nonzero_value_over_th_top = 0
        nonzero_value_over_th_bottom = 0
        nonzero_value_over_th_left = 0
        nonzero_value_over_th_right = 0

        # ratio = 0.35
        ratio = 0.5
        ratio_w_top = 0.45
        ratio_w_bottom = 0.6

        for n in range(gray.shape[0]):
            checking_top_nonzero_value = cv2.countNonZero(gray[n,:])

            if checking_top_nonzero_value > w * ratio_w_top:
                nonzero_value_over_th_top = n
                break

        for n in range(gray.shape[0]):
            checking_bottom_nonzero_value = cv2.countNonZero(gray[gray.shape[0]-1-n,:])

            if checking_bottom_nonzero_value > w * ratio_w_bottom:
                nonzero_value_over_th_bottom = gray.shape[0] - n
                break
        
        for n in range(gray.shape[1]):
            checking_left_nonzero_value = cv2.countNonZero(gray[:,n])

            if checking_left_nonzero_value > h * ratio:
                nonzero_value_over_th_left = n
                break
        
        for n in range(gray.shape[1]):
            checking_right_nonzero_value = cv2.countNonZero(gray[:,gray.shape[1]-1-n])

            if checking_right_nonzero_value > h * ratio:
                nonzero_value_over_th_right = gray.shape[1] - 1 - n
                break

        # x: 左端に画素がおおい
        # y: 上端に画素がおおい
        # w: 右端に画素がおおい
        # h: 下端に画素がおおい
        if not nonzero_value_over_th_left == 0: 
            def_x = abs(x - nonzero_value_over_th_left)
            x = nonzero_value_over_th_left
            w -= def_x
        
        if not nonzero_value_over_th_top == 0: 
            print("nonzero_value_over_th_topのなか")
            print("y =",y)
            print("nonzero_value_over_th_to =", nonzero_value_over_th_top)
            def_h = abs(y - nonzero_value_over_th_top)
            y = nonzero_value_over_th_top
            h -= def_h

        if not nonzero_value_over_th_right == gray.shape[1]:
            def_w = abs(x + w - nonzero_value_over_th_right )
            w -= def_w

        if not nonzero_value_over_th_bottom == gray.shape[0]: 
            print("nonzero_value_over_th_bottomのなか")
            print("h =",h)
            print("nonzero_value_over_th_bottom =", nonzero_value_over_th_bottom)
            def_h = abs(y + h - nonzero_value_over_th_bottom) 
            h -= def_h
        
        print("x,y,w,h =",x,y,w,h)


        # new_img = Image.new('RGB', (gray.shape[1], gray.shape[0]), (0, 0, 0))
        new_img = np.zeros((gray.shape[1], gray.shape[0], 3), np.uint8)
        img = np.array(new_img)
        # new_img = ImageDraw.Draw(im)
        img2 = cv2.rectangle(images,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        img_rectnagle = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=-1)
        img_rectnagle = cv2.cvtColor(img_rectnagle, cv2.COLOR_RGB2GRAY)

        top_nonzero_value = cv2.countNonZero(gray[0:20, 0:gray.shape[1]])
        top_edge_nonzero_value = cv2.countNonZero(gray[0:1, 0:gray.shape[1]])


        bottom_nonzero_value = \
        cv2.countNonZero(gray[gray.shape[0]-20:gray.shape[0], 0:gray.shape[1]])
        bottom_edge_nonzero_value = \
        cv2.countNonZero(gray[gray.shape[0]-1:gray.shape[0], 0:gray.shape[1]])
        

        left_nonzero_value_normal = cv2.countNonZero(gray[0:gray.shape[0], 0:20])

        left_nonzero_value = int(cv2.countNonZero(gray[0:int(gray.shape[0]*0.2), 0:20]) * 0.8)
        left_nonzero_value += int(cv2.countNonZero(gray[int(gray.shape[0]*0.2):int(gray.shape[0]*0.8), 0:20]) * 1.2)
        left_nonzero_value += int(cv2.countNonZero(gray[int(gray.shape[0]*0.8):gray.shape[0], 0:20]) * 0.8)

        left_edge_nonzero_value = cv2.countNonZero(gray[0:gray.shape[0], 0:1])
        
        right_nonzero_value_normal = \
        cv2.countNonZero(gray[0:gray.shape[0], gray.shape[1]-20:gray.shape[1]])

        right_nonzero_value = int( cv2.countNonZero(gray[0:int(gray.shape[0]*0.2), gray.shape[1]-20:gray.shape[1]]) * 0.8)
        right_nonzero_value += int( cv2.countNonZero(gray[int(gray.shape[0]*0.2):int(gray.shape[0]*0.8), gray.shape[1]-20:gray.shape[1]]) * 1.2 )
        right_nonzero_value += int(cv2.countNonZero(gray[int(gray.shape[0]*0.8):gray.shape[0], gray.shape[1]-20:gray.shape[1]]) * 0.8 )

        right_edge_nonzero_value = \
        cv2.countNonZero(gray[0:gray.shape[0], gray.shape[1]-1:gray.shape[1]])

        print("w =",gray.shape[1])
        print("h=",gray.shape[0])
        print("top_nonzero_value =",top_nonzero_value)
        print("bottom_nonzero_value =",bottom_nonzero_value)
        print("left_nonzero_value =",left_nonzero_value)
        print("right_nonzero_value =",right_nonzero_value)
        print("top_edge_nonzero_value =",top_edge_nonzero_value)
        print("bottom_edge_nonzero_value =",bottom_edge_nonzero_value)
        print("left_edge_nonzero_value =",left_edge_nonzero_value)
        print("right_edge_nonzero_value =",right_edge_nonzero_value)
        print("left_nonzero_value_normal =",left_nonzero_value_normal)
        print("right_nonzero_value_normal =",right_nonzero_value_normal)


        # top_rectnagle_nonzero_value = cv2.countNonZero(img_rectnagle[0:20, 0:x+w])

        # bottom_rectnagle_nonzero_value = \
        # cv2.countNonZero(img_rectnagle[img_rectnagle.shape[0]-20:img_rectnagle.shape[0], x:x+w])
        
        if x == 0 or x+w == gray.shape[1]-1:
            left_rectnagle_nonzero_value = cv2.countNonZero(gray[y:y+h, 0:20])
            
            right_rectnagle_nonzero_value = \
            cv2.countNonZero(gray[y:y+h, gray.shape[1]-20:gray.shape[1]])

            print(top_nonzero_value,bottom_nonzero_value,left_nonzero_value,right_nonzero_value)
            print(left_rectnagle_nonzero_value, right_rectnagle_nonzero_value)
            print(left_rectnagle_nonzero_value/(h*20), right_rectnagle_nonzero_value/(h*20))

            if left_rectnagle_nonzero_value/(h*20) > 0.8 or right_rectnagle_nonzero_value/(h*20) > 0.8:
                print("見切れている")
            else:
                print("見切れていない")

        else:
            print("計算しない")

        # 外接矩形の端から20px分の画素数の割合が6-8割以上だと見切れていると判断してみる？

        # 図形の描画
        cv2.circle(img2, (x_m,y_m), 10, color=(255, 0, 0), thickness=3)
        # cv2.rectangle(img2,(int(p1), 0), (int(p2), images.shape[1]), (255, 255, 255), thickness=2, lineType  =cv2.LINE_8, shift=0)

        # plt.imshow(gray)
        plt.imshow(img2, cmap="jet")
        plt.text(0, -50, "x = {}".format(x), fontsize=8)
        plt.text(0, -100, "y = {}".format(y), fontsize=8)
        plt.text(0, -150, "ratio_whitePixels = {}".format(ratio_whitePixels), fontsize=8)
        plt.axis("off")

        # ax = plt.subplot(1, 3, i + 3)
        # plt.imshow(img2, cmap="jet")
        # plt.text(0, -20, "whitePixels = {}".format(whitePixels), fontsize=8)
        # plt.text(0, -40, "ratio_whitePixels = {}".format(ratio_whitePixels), fontsize=8)
        # plt.text(0, -60, "front_back_flg = {}".format(front_back_flg), fontsize=8)
        # plt.axis("off")

plt.show()
