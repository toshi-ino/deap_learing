import os
import tensorflow as tf
from distutils.dir_util import copy_tree
import glob, shutil

num_skipped = 0
# ファイルをまとめたいフォルダを220404フォルダのtest_dataフォルダに保存する
path = "./test_data/ok_front"

# 新規で作成するファルダの名称
new_folder_name = "new_folder"

# バイクの種類のフォルダ
bike_types = os.listdir(path)
print(bike_types)
colors = ["black", "blue", "matblack", "other", "white", "silver","orange", "red", "green", "yellow"]

# フォルダを作成
new_directions = ["front", "back", "right", "left"]
for new_direction in (new_directions):
    for bike_type in (bike_types):
        if bike_type != ".DS_Store":
            new_dir_path = "./test_data/" + new_folder_name + "/" + bike_type
            os.makedirs(new_dir_path, exist_ok=True)

for bike_type in (bike_types):
    if bike_type != ".DS_Store":
        for color in (colors):
            # 特定のバイクやフォルダの場合
            if bike_type[0] == "_":
                from_folder_path = path + "/" + bike_type
            # デフォルトのバイクフォルダの場合
            else:
                from_folder_path = path + "/" + bike_type + "/" + color

            # 保存先のフォルダ
            to_folder_path = "./test_data/" + new_folder_name + "/" + bike_type

            if bike_type[0] == "_":
                copy_tree(from_folder_path, to_folder_path)

            else:
                # ファイルのリストを作成し、間引く数を設定
                files = glob.glob(from_folder_path + "/*")
                skipped_files = files[::20]
                # files_num = (sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR)))
                for f in skipped_files:
                    if os.path.exists(from_folder_path) == True:
                        shutil.copy(f, to_folder_path) 



