import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import seaborn as sns

def sliding_window(sequence, window_size, step_size):
    subsequences = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        subsequences.append(sequence[i:i + window_size])
    return np.array(subsequences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-data_name', type=str, default='energy', help="数据集名称")
    parser.add_argument('-target', type=str, default='T1', help="目标特征")
    parser.add_argument('-lambda_value', type=int, default=0.01, help="采样率")
    parser.add_argument('-mode', type=int, default=0, help="[0,1],不适用GPU、使用GPU")
    parser.add_argument('-url', type=str, default='http://10.12.54.122:5002/', help="服务器地址")
    args = parser.parse_args()
    folder_path = '../data' + '/' + args.data_name+".csv"
    data=pd.read_csv(folder_path)
    time_series=np.array(data[args.target])
    time_series=time_series[:200]
    subsequences=sliding_window(time_series, 10, 5)
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, label="Original Time Series", color="black", linewidth=2)  # 绘制原始时间序列

    # 绘制每个子序列的重叠部分
    for subseq in subsequences:
        plt.plot(range(len(subseq)), subseq, alpha=0.6)  # 使用 alpha 透明度以避免过于密集

    plt.title("Sliding Window Subsequence Overlap")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

