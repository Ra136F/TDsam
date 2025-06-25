import time

import matplotlib.pyplot as plt
import numpy as np

from util import data_loading, data_loading_aper, MinMaxScaler, merge_power_data

if __name__=='__main__':
    data_name="oil-well"
    folder_path = './data' + '/complete/BeanToCupCoffeemaker'
    # data, r_min, r_max = data_loading_aper(folder_path)
    data,r_min,r_max=merge_power_data(folder_path,"./data/power-1.csv")
    # cols=list(data.columns)
    # cols.remove('timestamp')
    # data,min,max=MinMaxScaler(np.array(data))
    # print(min,max)
    print(len(data))
    print(r_min)
    print(r_max)
    # data.to_csv("./data/power-1.csv",index=False,columns=data.columns)
