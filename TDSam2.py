import pandas as pd
import numpy as np
import time
import glob
import os

def TDSam(data,  r):
    # 从 DataFrame 提取指定列的数据作为列表
    D = data['P-TPT'].tolist()
    D_kp = [(D[0], D[-1])]  # 初始输出列表，包含第一个和最后一个元素

    for i in range(1, len(D) - 1):
        if D[i - 1] < D[i] > D[i + 1] or D[i - 1] > D[i] < D[i + 1]:  # 检查是否为非单调序列
            if D[i] > D[i + 1] + r and D[i] > D[i - 1] + r:
                D_kp.append(D[i])
            elif D[i] < D[i + 1] - r and D[i] < D[i - 1] - r:
                D_kp.append(D[i])
            elif abs(D[i + 1] - 2 * D[i] + D[i - 1]) > r:
                D_kp.append(D[i])

    return D_kp

# 指定 CSV 文件夹路径
folder_path = './0'

# 使用 glob 找到所有 CSV 文件路径
all_files = glob.glob(os.path.join(folder_path, "*.csv"))
columns_to_read = ['P-TPT']

# 用列表推导式读取每个文件并存入 DataFrame 列表
df_list = [pd.read_csv(file,usecols=columns_to_read) for file in all_files]

# 将所有 DataFrame 进行合并
df_combined = pd.concat(df_list, ignore_index=True)
# diff=df_combined['P-TPT'].max()-df_combined['P-TPT'].min()
# adjacent_diffs =df_combined['P-TPT'].diff().abs().iloc[1:]
# min_adjacent_diff = adjacent_diffs[adjacent_diffs != 0].min()
start_time = time.time()
D_kp=TDSam(df_combined,10)
end_time = time.time()
execution_time = end_time - start_time
print(f"TDSam 算法执行时间: {execution_time:.6f} 秒")
print(len(D_kp))
print(len(df_combined))