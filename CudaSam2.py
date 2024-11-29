from numba import cuda
import pandas as pd
import time
import numpy as np
import glob
import os

@cuda.jit
def TDSam_cuda(D, r, D_kp, n):
    i = cuda.grid(1)
    if i == 0:
        # 将第一个点添加到关键点
        D_kp[0] = D[0]
    elif i == n - 1:
        # 将最后一个点添加到关键点
        D_kp[n - 1] = D[n - 1]
    elif 1 <= i < n - 1:
        # 判断是否为极大值或极小值
        if D[i - 1] < D[i] > D[i + 1] or D[i - 1] > D[i] < D[i + 1]:
            if D[i] > D[i + 1] + r and D[i] > D[i - 1] + r:
                D_kp[i] = D[i]
            elif D[i] < D[i + 1] - r and D[i] < D[i - 1] - r:
                D_kp[i] = D[i]
        elif abs(D[i + 1] - 2 * D[i] + D[i - 1]) > r:
            D_kp[i] = D[i]
        else:
            D_kp[i] = 0  # 不符合条件，填充为 0（可以视为无效点）

folder_path = './0'
all_files = glob.glob(os.path.join(folder_path, "*.csv"))
columns_to_read = ['timestamp','P-TPT']
df_list = [pd.read_csv(file,usecols=columns_to_read) for file in all_files]
df_combined = pd.concat(df_list, ignore_index=True)
# print(df_combined.head())
D = df_combined['P-TPT'].values
timestamps = df_combined['timestamp'].values  # 保留时间戳列
# print(D)
n = len(D)
D_kp = np.zeros_like(D)
threads_per_block = 1024
blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block


start_time = time.time()
D_device = cuda.to_device(D)
D_kp_device = cuda.to_device(D_kp)
TDSam_cuda[blocks_per_grid, threads_per_block](D_device, 10, D_kp_device, n)
cuda.synchronize()
D_kp=D_kp_device.copy_to_host()
end_time = time.time()

execution_time = end_time - start_time

print(f"TDSam CUDA 加速算法执行时间: {execution_time:.6f} 秒")

# 将非零结果转换为 DataFrame 并显示
result_df = pd.DataFrame({
    'timestamp': timestamps[D_kp != 0],  # 保留 timestamp 对应非零 P-TPT
    'P-TPT': D_kp[D_kp != 0]  # 只保留非零的 P-TPT 关键点
})
# print(len(D_kp)
print(f'原始数据条数:{len(df_combined)}')
print(f'采样后数据条数:{len(result_df)}')
