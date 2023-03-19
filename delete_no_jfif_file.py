# leftのフォルダ
# _shallow_diagonalの中に変なファイルがいるので削除すること
# rightのフォルダ
# _shallow_diagonal、_right_turnned_wheelの中に変なファイルがいるので削除すること
# jpg、webpのファイルも削除すること

import os
import tensorflow as tf

num_skipped = 0
path = "./png"
image_folders = os.listdir(path)
print(image_folders) 

for folder_name in (image_folders):
    if not folder_name.startswith('.'): 
        for current_dir, sub_dirs, files_list in os.walk(path + "/" + folder_name):

            # print(u"現在のディレクトリは {} です".format(current_dir)) 
            # print(u"サブディレクトリは {} です".format(sub_dirs)) 
            # print(u"ディレクトリ内のファイルは {} です".format(files_list)) 
            # print("//////////////////////////////////////////////")

            for file_name in files_list:
                fpath = os.path.join(current_dir,file_name)
                try:
                    fobj = open(fpath, "rb")
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
                finally:
                    fobj.close()

                if not is_jfif:
                    num_skipped += 1
                    # Delete corrupted image
                    os.remove(fpath)

print("Deleted %d images" % num_skipped)
