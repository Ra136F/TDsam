import time

import matplotlib.pyplot as plt
import numpy as np

from util import data_loading, data_loading_aper, MinMaxScaler

if __name__=='__main__':
    data_name="oil-well"
    folder_path = './data' + '/data/'
    data, r_min, r_max = data_loading_aper(folder_path)
    # cols=list(data.columns)
    # cols.remove('timestamp')
    # data,min,max=MinMaxScaler(np.array(data))
    # print(min,max)
    print(len(data))
    print(r_min)
    print(r_max)
    data.to_csv("./data/oil-well-1.csv",index=False,columns=data.columns)
