from pathlib import Path
import os
import glob, shutil


def get_paths(input_dir, exts=None):
    paths = sorted([x for x in input_dir.glob("**/*")])
    if exts:
        paths = list(filter(lambda x: x.suffix in exts, paths))

    return paths

dirs = ["cutoff", "extend", "50-100"]

for dir in dirs:
    # ディレクトリ内の指定した拡張子のファイルをすべて取得する。
    input_dir = Path("./test_data/ng_cutoff_50-100_right/" + dir)

    # 新しいフォルダを作成する
    new_dir_path = "./test_data/new_folder/" + dir
    os.makedirs(new_dir_path, exist_ok=True)

    i = 0
    for f_path in get_paths(input_dir, exts=[".jpg", ".jpeg", ".png"]):
        if (i % 10) == 0: 
            print(f_path)
            shutil.copy(f_path, new_dir_path) 
        i = i + 1

