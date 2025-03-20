import glob
import os

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dtw_distance(seq1, seq2):
    """
    计算两个序列之间的 DTW 距离
    :param seq1: 序列 1，数组形式
    :param seq2: 序列 2，数组形式
    :return: DTW 距离
    """
    len1, len2 = len(seq1), len(seq2)
    dtw_matrix = np.zeros((len1 + 1, len2 + 1))

    # 初始化 DTW 矩阵
    dtw_matrix[0, :] = float('inf')
    dtw_matrix[:, 0] = float('inf')
    dtw_matrix[0, 0] = 0

    # 动态规划填充 DTW 矩阵
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            # 计算最小的累积距离
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # 插入
                dtw_matrix[i, j - 1],    # 删除
                dtw_matrix[i - 1, j - 1] # 匹配
            )

    # 返回最终的 DTW 距离
    return dtw_matrix[len1, len2]



def data_loading(folder_path,target):
    all_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
    # 读取所有文件并合并
    df_list = [pd.read_csv(file).fillna(0) for file in all_files]
    df_combined = pd.concat(df_list, ignore_index=True)
    print(f"原始数据条数: {len(df_combined)}")
    D = df_combined[target].values
    min,max = calculate_gap(D)
    print(f'最大是:{max},最小是{min}')
    return df_combined,min,max

#寻找相邻数据点的最小、最大差值
def calculate_gap(data):
    data = np.asarray(data)
    data = np.nan_to_num(data, nan=0.0)
    # 计算相邻数据点的差值
    differences = np.diff(data)
    # 取绝对值，忽略方向
    differences = np.abs(differences)
    non_zero_differences = differences[differences > 0]
    # 检查是否有非零差值
    if non_zero_differences.size == 0:
        raise ValueError("时间序列数据的所有相邻差值均为 0，无法计算 δ 值")
    min = np.min(non_zero_differences)
    max = np.max(non_zero_differences)
    return min,max


