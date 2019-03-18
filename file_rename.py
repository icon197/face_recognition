import shutil
import os

path = "/home/thanh/mayhocnangcao/baocao/imggoc/result2/"

couter = 0
for file_name in os.listdir(path):
    path_file = "{}/{}".format(path, file_name)
    path_file_res = "{}/new_{}".format(path, file_name)
    if os.path.isfile(path_file):
        os.rename(path_file, path_file_res)