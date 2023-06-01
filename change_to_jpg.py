#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conda install pillow
import os, glob
from PIL import Image
# def main():
#     filepath_list = glob.glob(input_path + '/*.png') # .pngファイルをリストで取得する
#     for filepath in filepath_list:
#         basename  = os.path.basename(filepath) # ファイルパスからファイル名を取得
#         save_filepath = out_path + '/' + basename [:-4] + '.jpg' # 保存ファイルパスを作成
#         img = Image.open(filepath)
#         img = img.convert('RGB') # RGBA(png)→RGB(jpg)へ変換
#         img.save(save_filepath, "JPEG", quality=95)
#         print(filepath, '->', save_filepath)
#         if flag_delete_original_files:
#             os.remove(filepath)
#             print('delete', filepath)

# if __name__ == '__main__':
#     input_path = './test_data' # オリジナルpngファイルがあるフォルダを指定
#     out_path = input_path # 変換先のフォルダを指定
#     flag_delete_original_files = False # 元ファイルを削除する場合は、True指定

#     main()


num_skipped = 0
path = "./training_data"
image_folders = os.listdir(path)
print(image_folders) 
out_path = "./changed_training_data" # 変換先のフォルダを指定

for folder_name in (image_folders):
    print(folder_name)
    if not folder_name.startswith('.'): 
        for current_dir, sub_dirs, files_list in os.walk(path + "/" + folder_name):

            for file_name in files_list:
                if not file_name.startswith('.'): 
                    fpath = os.path.join(current_dir,file_name)
                    img = Image.open(fpath)
                    img = img.convert('RGB') # RGBA(png)→RGB(jpg)へ変換
                    basename  = os.path.basename(fpath) # ファイルパスからファイル名を取得
                    save_filepath = out_path + '/' + folder_name + "/" + basename [:-4] + '.jpg' # 保存ファイルパスを作成
                    img.save(save_filepath, "JPEG", quality=95)

