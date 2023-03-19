import os
import tensorflow as tf
from distutils.dir_util import copy_tree

num_skipped = 0
directions = ["前", "後", "右", "左"]
# ファイルをまとめたいフォルダを220404フォルダのtest_dataフォルダに保存する
path = "./test_data/ok_right"

# バイクの種類のフォルダ
bike_types = os.listdir(path)
print(bike_types)
colors = ["black", "bule", "madblack", "other", "white", "silver","orange", "red", "green", "yellow"]


# フォルダを作成
new_directions = ["front", "back", "right", "left"]
for new_direction in (new_directions):
    for bike_type in (bike_types):

        if bike_type != ".DS_Store":
            new_dir_path = path + "/" + new_direction + "/" + bike_type 
            os.makedirs(new_dir_path)

for bike_type in (bike_types):

    if bike_type != ".DS_Store":
        # フォルダを作成
        # for new_direction in (new_directions):
        #     new_dir_path = path + "/" + bike_type + "/" + new_direction
        #     os.mkdir(new_dir_path)

        for color in (colors):
            for direction_number in range(0,3):
                from_folder_path = path + "/" + bike_type + "/" + color + "/" + directions[direction_number]
                to_folder_path = path + "/" + new_directions[direction_number] + "/" + bike_type

                if os.path.exists(from_folder_path) == True:
                    copy_tree(from_folder_path, to_folder_path)

