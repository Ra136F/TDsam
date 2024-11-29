import pandas as pd
import numpy as np
from numba import njit, prange
import glob
import os
import time


@njit(parallel=True)
def TDSam_SIMD(D, r):
    n = len(D)
    D_kp = np.zeros(n, dtype=D.dtype)  # 初始化关键点数组

    # 遍历数据点，使用并行化和 SIMD 指令
    for i in prange(1, n - 1):  # 使用 prange 使得循环并行化
        # 判断是否为局部极大值或极小值
        if (D[i - 1] < D[i] > D[i + 1]) or (D[i - 1] > D[i] < D[i + 1]):
            if D[i] > D[i + 1] + r and D[i] > D[i - 1] + r:
                D_kp[i] = D[i]
            elif D[i] < D[i + 1] - r and D[i] < D[i - 1] - r:
                D_kp[i] = D[i]
        elif abs(D[i + 1] - 2 * D[i] + D[i - 1]) > r:
            D_kp[i] = D[i]

    # 保留第一个和最后一个数据点
    D_kp[0] = D[0]
    D_kp[n - 1] = D[n - 1]

    return D_kp[D_kp != 0]  # 返回非零关键点


# 读取 CSV 文件夹并合并
folder_path = './0'
all_files = glob.glob(os.path.join(folder_path, "*.csv"))
columns_to_read = ['P-TPT']
df_list = [pd.read_csv(file, usecols=columns_to_read) for file in all_files]
df_combined = pd.concat(df_list, ignore_index=True)

# 将数据转换为 NumPy 数组并初始化参数
D = df_combined['P-TPT'].values
r = 10

# 执行 TDSam SIMD 优化算法并记录时间
start_time = time.time()
D_kp = TDSam_SIMD(D, r)
end_time = time.time()
execution_time = end_time - start_time

print(f"TDSam SIMD 优化算法执行时间: {execution_time:.6f} 秒")
print(f"关键点数量: {len(D_kp)}")
print(f"原始数据长度: {len(D)}")
