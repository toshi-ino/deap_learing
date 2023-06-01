# 参考URL
# https://pystyle.info/opencv-histogram/

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# 画像を読み込む。
files = glob.glob("./test_data/data/*")
for file in files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 白黒反転作業作業
    img=255-img

# img = cv2.resize(img, dsize=(512, 512))
# height, width, rgb = img.shape
# size = height * width

# print("height =",height)
# print("width =",width)
# print("size =",size)

# 閾値の設定
threshold = 170

# 二値化(閾値100を超えた画素を255にする。)
ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

# 二値化画像の表示
cv2.imshow("img_th", img_thresh)
cv2.waitKey()
cv2.destroyAllWindows()


# ヒストグラムを作成する。
n_bins = 256  # ビンの数
hist_range = [0, 256]  # 集計範囲

hists = []
# channels = {0: "blue", 1: "green", 2: "red"}
channels = {0: "gray"}
for ch in channels:
    hist = cv2.calcHist(
        [img], channels=[ch], mask=None, histSize=[n_bins], ranges=hist_range
    )
    hist = hist.squeeze(axis=-1)
    hists.append(hist)


# 描画する。
def plot_hist(bins, hist, color):
    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)
    ax.bar(centers, hist, width=widths, color=color)


bins = np.linspace(*hist_range, n_bins + 1)

plt.xticks(color = "None")
plt.yticks(color = "None")
plt.tick_params(bottom = False, left = False)
plt.imshow(img)

fig, ax = plt.subplots()
ax.set_xticks([0, 256])
ax.set_xlim([0, 256])
ax.set_xlabel("Pixel Value")
ax.set_ylim([0,300000])
for hist, color in zip(hists, channels.values()):
    plot_hist(bins, hist, color=color)
plt.show()


