import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# 画像を読み込む。
# img = cv2.imread("sample.jpg")

files = glob.glob("./test_data/test_data_16/*")
for file in files:
    img = cv2.imread(file)

img = cv2.resize(img, dsize=(512, 512))
height, width, rgb = img.shape
size = height * width

# HSV に変換する。
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_hsv = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2HSV_FULL)
img_h, img_s, img_v = cv2.split(img_hsv)
imgs_hsv = [img_h, img_s, img_v]
imgs_type = {0:"H", 1:"S", 2:"V"}

# 2次元ヒストグラムを作成する。
hist_range1 = [0, 256]
hist_range2 = [0, 256]
n_bins1, n_bins2 = 20, 20

hists = []
channel_pairs = [[0, 1], [1, 2], [0, 2]]
for pair in channel_pairs:
    hist = cv2.calcHist(
        [hsv], channels=pair, mask=None, histSize=[20, 20], ranges=[0, 256, 0, 256]
    )
    hists.append(hist)

# 描画する。
ch_names = {0: "Hue", 1: "Saturation", 2: "Brightness"}
channels = {0, 1, 2}

fig = plt.figure(figsize=(10, 8))
for i, (hist, ch) in enumerate(zip(hists, channels), 1):
    ax = fig.add_subplot(2, 3, i+1)
    plt.xticks(color = "None")
    plt.yticks(color = "None")
    plt.tick_params(bottom = False, left = False)
    plt.title(imgs_type[i-1])
    ax.imshow(255-imgs_hsv[i-1], cmap='gray')

    pair = channel_pairs[i-1]
    xlabel, ylabel = ch_names[pair[0]], ch_names[pair[1]]

    ax = fig.add_subplot(2, 3, i+3)
    fig.subplots_adjust(wspace=0.3)
    ax.imshow(hist, cmap="jet")

    # 2Dヒストグラムを描画する。
    ax.set_title(f"{xlabel} and {ylabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

plt.show()