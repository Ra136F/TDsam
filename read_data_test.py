import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import data_loading, MinMaxScaler, calculate_gap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-data_name', type=str, default='gas2', help="数据集名称")
    parser.add_argument('-target', type=str, default='sensor_1', help="目标特征")
    parser.add_argument('-lambda_value', type=int, default=0.01, help="采样率")
    parser.add_argument('-mode', type=int, default=0, help="[0,1],不适用GPU、使用GPU")
    parser.add_argument('-url', type=str, default='http://10.12.54.122:5002/', help="服务器地址")
    args = parser.parse_args()
    folder_path = './data' + '/' + args.data_name+".csv"
    # data, r_min, r_max = data_loading(folder_path, args.target)

    data=pd.read_csv(folder_path)
    # data=data[:int(len(data)*0.1)]
    min, max = calculate_gap(data[args.target].values)
    print(min, max)
    print(f"长度:{len(data)}")
    # data,min,max=MinMaxScaler(np.array(data))
    # print(min.shape)
    plt.plot( data[[args.target]], label='Original Data')
    plt.legend()
    plt.show()
    # data.to_csv("./data/oil-well-1.csv", index=False, columns=data.columns)

