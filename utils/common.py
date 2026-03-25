import datetime
import numpy as np
import os
from os.path import exists
import shutil
import pandas as pd


def get_unique_num():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + str(np.random.randint(10, 100))


def get_print_time(t):
    h = int(t // 3600)
    m = int((t // 60) % 60)
    s = int(t % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def clear_dir(dir):
    if exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)




def load_data(path, names):
    xl = pd.ExcelFile(path)
    real_data_list = []
    for epo in names:
        real_data_list.append(xl.parse(f'{epo}').values)
    return real_data_list
