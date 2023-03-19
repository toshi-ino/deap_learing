# 参考URL
# https://pystyle.info/opencv-histogram/

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import math


# 画像の読み込み
files = glob.glob("./test_data/test_data_16/*")
for file in files:
    img = cv2.imread(file)
    gray = cv2.imread(file, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, dsize=(512, 512))
    gray = cv2.resize(gray, dsize=(512, 512))

# RGBをHSVに変換
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
h, s, v = cv2.split(hsv)

# look_up_table = np.zeros((256,1), dtype = 'uint8')
# for i in range(256):
#     look_up_table[i][0] = 255 * (np.sin(np.pi * (i/255 - 1/2)) + 1) / 2

# v = cv2.LUT(v,look_up_table)
gamma = 1.25       
gamma_cvt = np.zeros((256,1),dtype = 'uint8')
for i in range(256):
    gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)

v = cv2.LUT(v,gamma_cvt)

# 背景の画素をヒストグラムに含めないためのマスク画像を作成
ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
thresh1 = cv2.bitwise_not(thresh1)
thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)

# # 読み込んだ画像とマスク画像を合成
# img_AND = cv2.bitwise_and(gray, thresh1)

# エッジ検出
# gray = cv2.bitwise_not(gray)
img_edge = cv2.Canny(v, 40, 420)

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.xticks(color = "None")
plt.yticks(color = "None")
fig.add_subplot(1, 2, 2)
plt.imshow(img_edge, cmap='gray')
plt.xticks(color = "None")
plt.yticks(color = "None")
plt.show()


# 画像を分割
i_y = 0
i_multi_x = 0
for i in range(256):
    ax = plt.subplot(16, 16, i + 1)
    plt.xticks(color = "None")
    plt.yticks(color = "None")
    plt.tick_params(bottom = False, left = False)

    x_start = 32 * (i - (i_multi_x * 16))
    x_end = x_start + 32

    y_start = 32 * i_y
    y_end = y_start + 32

    if x_end == 512:
        i_y = i_y + 1
        i_multi_x = i_multi_x + 1

    # gray_rgb_laplacian = cv2.Laplacian(gray[y_start:y_end, x_start:x_end], cv2.CV_32F, ksize=1).var() 
    # 分割した画像のグレースケール値の平均を算出
    # v_mean = v[y_start:y_end, x_start:x_end].mean()

    # 分割した画像のグレースケール値の平均を算出
    mask_edge_value = cv2.countNonZero(thresh1[y_start:y_end, x_start:x_end])
    # edge_value = cv2.countNonZero(thresh1[y_start:y_end, x_start:x_end])
    plt.imshow(img_edge[y_start:y_end, x_start:x_end], cmap='gray')   

    if mask_edge_value < 924: 
        edge_value = cv2.countNonZero(img_edge[y_start:y_end, x_start:x_end])

        if edge_value < 20:
            plt.text(0, -1, edge_value, fontsize=6, color="red", fontweight="bold")
        else:
            plt.text(0, -1, edge_value, fontsize=6)


# gray_mean, std = cv2.meanStdDev(img, mask=thresh1)
# and_mean, and_std = cv2.meanStdDev(img_AND)
# v_mean, v_std = cv2.meanStdDev(v)
# v_mean_rbg, v_std_rbg = cv2.meanStdDev(base_v)
# print("gray_mean =",gray_mean)
# print("gray_std =",std)
# print("v_mean =",v_mean)
# print("v_std =",v_std)
# print("v_and_mean =",and_mean)
# print("v_and_std =",and_std)

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()