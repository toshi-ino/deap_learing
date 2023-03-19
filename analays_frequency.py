# 参考にしたURL
# python+opencvで画像処理の勉強4 周波数領域におけるフィルタリング
# https://qiita.com/tanaka_benkyo/items/bfa35e7f08faa7b7a985

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


# 画像をグレースケール形式で読み込む。
files = glob.glob("./test_data/test_data_16/*")
imgs = []
for file in files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(512, 512))
    height, width = img.shape
    size = height * width

    print("height =",height)
    print("width =",width)
    print("size =",size)
    imgs.append(img)

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for i in range(3):
    dft = cv2.dft(np.float32(imgs[i]), flags = cv2.DFT_COMPLEX_OUTPUT)
    # ゼロ周波数の成分を中心に移動
    dft_shift = np.fft.fftshift(dft)
    # パワースペクトル
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    """
    numpyによるフーリエ変換
    f = np.fft.fft2(imgs[i])
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    """
    ax[0][i].imshow(imgs[i], 'gray')
    ax[0][i].set_xticks([])
    ax[0][i].set_yticks([])

    ax[1][i].imshow(magnitude_spectrum, 'gray')
    ax[1][i].set_xticks([])
    ax[1][i].set_yticks([])
plt.show()