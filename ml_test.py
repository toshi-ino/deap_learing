"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""
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
##
## モデルの学習を行う: learn_model_flg 
## 学習したモデルを保存する: save_model_flg *モデルの保存は learn_model_flg と save_model_flg が True の時に行う
## 学習済みモデルを使用する: use_learned_model_flg
## モデルの評価を行う: evaluate_trained_model_flg
## モデルの追加学習を行う: use_learned_model_for_learning_flg
##
"""
# ###################################################################################
learn_model_flg = False
save_model_flg = False
use_learned_model_flg = True
evaluate_trained_model_flg =  True
use_learned_model_for_learning_flg = False
# ###################################################################################


"""
## データの読み込み
"""
# Model / data parameters
num_classes = 2
input_shape = (256, 256, 3)

train_ds = image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
validation_ds = image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))

# image_dataset_from_directoryのshuffleをTrueにするとデータをシャッフルするが、test_dsを使用する度にデータをシャッフルするため、
# 正解データのラベルとデータを分けたあとにtest_dsを呼び出すと、データとラベルの関係が崩れるので要注意
test_ds = image_dataset_from_directory(
    directory='test_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256),
    shuffle=False,
    validation_split=None
    )

"""
## モデルの設定
"""
if learn_model_flg:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),  
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),  
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(40, activation="relu",kernel_initializer='he_normal'),
            layers.Dense(20, activation="relu",kernel_initializer='he_normal'),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    """
    ## モデルのトレーニング
    """
    batch_size = 64
    epochs = 15
    if use_learned_model_for_learning_flg:
        model = load_model('learned_model.h5')
    else:
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy","Precision","AUC"])

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", verbose=0, patience=5, restore_best_weights=True)
    # model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=validation_ds, callbacks=[callback])
    model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=validation_ds)
    # model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=validation_ds)


"""
## 学習したモデルを保存する
"""
if learn_model_flg & save_model_flg:
    model.save('learned_model.h5')
    print("")
    print("finish saving the model!")
    print("")

"""
## 学習済みモデルを読み込む
## 学習済みモデルはfile.pyと同じファイルに保存すること
"""
if use_learned_model_flg:
    model = load_model('learned_model.h5')

"""
## Evaluate the trained model
## testデータを使用して精度の確認をおこなう
"""
if evaluate_trained_model_flg:
    # evaluate_trained_model(test_ds, model)
        # testデータを使用して精度の確認をおこなう
    score = model.evaluate(test_ds, verbose=0)
    print("@@@@@@@@@@@@@@@@@ Result Test @@@@@@@@@@@@@@@")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Test Precision:", score[2])
    print("Test AUC:", score[3])
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    if True:
        test_np = tfds.as_numpy(test_ds)
        y_preds = np.empty((2,2), int)
        y_pred_argmax_datas = np.empty(0, int)
        y_test_argmax_datas = np.empty(0, int)
        for i in range(len(test_np)):
            print(i)

            # y_test: テストのラベル
            # x_test: テストの画像データ
            # 画像データの形 [(128, 128, 3, 32), (128, 128, 3, 32), (128, 128, 3, 32)]
            # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
            y_test = [x[1] for x in test_np][i]
            x_test = [x[0] for x in test_np][i]

            threshold = 0.5
            y_pred_prob = model.predict(x_test)
            y_pred = (y_pred_prob[:,1] >= threshold).astype(int)
            # y_pred_argmax = tf.argmax(y_pred, axis = 1).numpy()
            y_test_argmax = tf.argmax(y_test, axis = 1).numpy()

            if i == 0:
                y_preds = y_pred
                # y_pred_argmax_datas = y_pred_argmax
                y_pred_prob_datas = y_pred_prob
                y_pred_argmax_datas = y_pred
                y_test_argmax_datas = y_test_argmax
            else:
                y_preds = np.append(y_preds, y_pred, axis=0)
                # y_pred_argmax_datas = np.append(y_pred_argmax_datas,y_pred_argmax)
                y_pred_prob_datas = np.append(y_pred_prob_datas,y_pred_prob,axis=0)
                y_pred_argmax_datas = np.append(y_pred_argmax_datas,y_pred)
                y_test_argmax_datas = np.append(y_test_argmax_datas,y_test_argmax)

        print("@@@@@@@@@@@@@@@@@ threshold @@@@@@@@@@@@@@@")
        print(threshold)
        print("")

        # モデルによる予測値
        print("@@@@@@@@@@@@@@@@@ y_pred_argmax_datas @@@@@@@@@@@@@@@")
        print(y_pred_argmax_datas)
        print("")

            # テストデータのクラス
        print("@@@@@@@@@@@@@@@@@ y_test_argmax_datas @@@@@@@@@@@@@@@")
        print(y_test_argmax_datas)
        print("")

        # 個別の予測値とテストデータの表示
        print("@@@@@@@@@@@@@@@@@ y_pred_argmax_data @@@@@@@@@@@@@@@")

        class1 = 0
        class2 = 0
        for i in range(len(y_pred_argmax_datas)):
            if y_test_argmax_datas[i] == 0:
                class1 += 1
                # print("class1",class1, "  ", y_pred_argmax_datas[i], "  ", y_preds[i])
                print("class1",class1, "  ", y_pred_argmax_datas[i], "  ", y_test_argmax_datas[i], "  ",y_pred_prob_datas[i])
            elif y_test_argmax_datas[i] == 1:
                class2 += 1
                # print("class2",class2, "  ", y_pred_argmax_datas[i], "  ", y_preds[i])
                print("class2",class2, "  ", y_pred_argmax_datas[i], "  ", y_test_argmax_datas[i], "  ",y_pred_prob_datas[i])

        print("")
        print("@@@@@@@@@@@@@@@@@ Result matrix @@@@@@@@@@@@@@@")
        print(tf.math.confusion_matrix(y_test_argmax_datas, y_pred_argmax_datas))



# def evaluate_trained_model(test_ds, model):
#     # testデータを使用して精度の確認をおこなう
#     score = model.evaluate(test_ds, verbose=0)
#     print("@@@@@@@@@@@@@@@@@ Result Test @@@@@@@@@@@@@@@")
#     print("Test loss:", score[0])
#     print("Test accuracy:", score[1])
#     print("Test Precision:", score[2])
#     print("Test AUC:", score[3])
#     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

#     if True:
#         confusion_matrix(test_ds, model)


# """
# ## コンフュージョンマトリクス
# """
# def confusion_matrix(test_ds, model):
#     test_np = tfds.as_numpy(test_ds)
#     y_preds = np.empty((2,2), int)
#     y_pred_argmax_datas = np.empty(0, int)
#     y_test_argmax_datas = np.empty(0, int)
#     for i in range(len(test_np)):
#         print(i)

#         # y_test: テストのラベル
#         # x_test: テストの画像データ
#         # 画像データの形 [(128, 128, 3, 32), (128, 128, 3, 32), (128, 128, 3, 32)]
#         # https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
#         y_test = [x[1] for x in test_np][i]
#         x_test = [x[0] for x in test_np][i]

#         threshold = 0.5
#         y_pred_prob = model.predict(x_test)
#         y_pred = (y_pred_prob[:,1] >= threshold).astype(int)
#         # y_pred_argmax = tf.argmax(y_pred, axis = 1).numpy()
#         y_test_argmax = tf.argmax(y_test, axis = 1).numpy()

#         if i == 0:
#             y_preds = y_pred
#             # y_pred_argmax_datas = y_pred_argmax
#             y_pred_prob_datas = y_pred_prob
#             y_pred_argmax_datas = y_pred
#             y_test_argmax_datas = y_test_argmax
#         else:
#             y_preds = np.append(y_preds, y_pred, axis=0)
#             # y_pred_argmax_datas = np.append(y_pred_argmax_datas,y_pred_argmax)
#             y_pred_prob_datas = np.append(y_pred_prob_datas,y_pred_prob,axis=0)
#             y_pred_argmax_datas = np.append(y_pred_argmax_datas,y_pred)
#             y_test_argmax_datas = np.append(y_test_argmax_datas,y_test_argmax)

#     print("@@@@@@@@@@@@@@@@@ threshold @@@@@@@@@@@@@@@")
#     print(threshold)
#     print("")

#     # モデルによる予測値
#     print("@@@@@@@@@@@@@@@@@ y_pred_argmax_datas @@@@@@@@@@@@@@@")
#     print(y_pred_argmax_datas)
#     print("")

#         # テストデータのクラス
#     print("@@@@@@@@@@@@@@@@@ y_test_argmax_datas @@@@@@@@@@@@@@@")
#     print(y_test_argmax_datas)
#     print("")

#     # 個別の予測値とテストデータの表示
#     print("@@@@@@@@@@@@@@@@@ y_pred_argmax_data @@@@@@@@@@@@@@@")

#     class1 = 0
#     class2 = 0
#     for i in range(len(y_pred_argmax_datas)):
#         if y_test_argmax_datas[i] == 0:
#             class1 += 1
#             # print("class1",class1, "  ", y_pred_argmax_datas[i], "  ", y_preds[i])
#             print("class1",class1, "  ", y_pred_argmax_datas[i], "  ", y_test_argmax_datas[i], "  ",y_pred_prob_datas[i])
#         elif y_test_argmax_datas[i] == 1:
#             class2 += 1
#             # print("class2",class2, "  ", y_pred_argmax_datas[i], "  ", y_preds[i])
#             print("class2",class2, "  ", y_pred_argmax_datas[i], "  ", y_test_argmax_datas[i], "  ",y_pred_prob_datas[i])

#     print("")
#     print("@@@@@@@@@@@@@@@@@ Result matrix @@@@@@@@@@@@@@@")
#     print(tf.math.confusion_matrix(y_test_argmax_datas, y_pred_argmax_datas))