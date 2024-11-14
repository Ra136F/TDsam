import numpy as np
from numba import njit, prange
import pandas as pd
import os
import time
import glob

@njit(parallel=True)
def TDSam_numba(D, r):
    n = len(D)
    D_kp = np.zeros(n, dtype=np.float32)

    for i in prange(1, n - 1):  # prange 用于并行化循环
        # 判断是否为局部极大值或极小值
        if (D[i - 1] < D[i] > D[i + 1]) or (D[i - 1] > D[i] < D[i + 1]):
            if D[i] > D[i + 1] + r and D[i] > D[i - 1] + r:
                D_kp[i] = D[i]
            elif D[i] < D[i + 1] - r and D[i] < D[i - 1] - r:
                D_kp[i] = D[i]
            elif abs(D[i + 1] - 2 * D[i] + D[i - 1]) > r:
                D_kp[i] = D[i]

    return D_kp

folder_path = './0'
all_files = glob.glob(os.path.join(folder_path, "*.csv"))
columns_to_read = ['P-TPT']
df_list = [pd.read_csv(file,usecols=columns_to_read) for file in all_files]
df_combined = pd.concat(df_list, ignore_index=True)

D = df_combined['P-TPT'].values
n = len(D)
D_kp = np.zeros_like(D)
threads_per_block = 32
blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
