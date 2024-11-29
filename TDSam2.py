import pandas as pd
import numpy as np
import time
import glob
import os

def TDSam(D,  r,D_kp,n):
    # 从 DataFrame 提取指定列的数据作为列表

    for i in range(n):
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

# 指定 CSV 文件夹路径
folder_path = './0'

# 使用 glob 找到所有 CSV 文件路径
all_files = glob.glob(os.path.join(folder_path, "*.csv"))
columns_to_read = ['timestamp','P-TPT']

# 用列表推导式读取每个文件并存入 DataFrame 列表
df_list = [pd.read_csv(file,usecols=columns_to_read) for file in all_files]
# 将所有 DataFrame 进行合并
df_combined = pd.concat(df_list, ignore_index=True)
D = df_combined['P-TPT'].values
timestamps = df_combined['timestamp'].values  # 保留时间戳列
n = len(D)
D_kp = np.zeros_like(D)
# diff=df_combined['P-TPT'].max()-df_combined['P-TPT'].min()
# adjacent_diffs =df_combined['P-TPT'].diff().abs().iloc[1:]
# min_adjacent_diff = adjacent_diffs[adjacent_diffs != 0].min()
start_time = time.time()
TDSam(D, 10, D_kp, n)
end_time = time.time()
execution_time = end_time - start_time
print(f"TDSam 算法执行时间: {execution_time:.6f} 秒")
result_df = pd.DataFrame({
    'timestamp': timestamps[D_kp != 0],  # 保留 timestamp 对应非零 P-TPT
    'P-TPT': D_kp[D_kp != 0]  # 只保留非零的 P-TPT 关键点
})
print(result_df)

# print(len(df_combined))