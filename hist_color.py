# 参考URL
# https://pystyle.info/opencv-histogram/

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# 画像を読み込む。
files = glob.glob("./test_data/test_data_16/*")
for file in files:
    img = cv2.imread(file)

img = cv2.resize(img, dsize=(512, 512))
height, width, rgb = img.shape
size = height * width

print("height =",height)
print("width =",width)
print("size =",size)


# ヒストグラムを作成する。
n_bins = 256  # ビンの数
hist_range = [0, 256]  # 集計範囲

hists = []
channels = {0: "blue", 1: "green", 2: "red"}
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
ax.set_ylim([0,10000])
for hist, color in zip(hists, channels.values()):
    plot_hist(bins, hist, color=color)
plt.show()


