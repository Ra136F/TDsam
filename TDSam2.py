import pandas as pd
import numpy as np
import time
import glob
import os



def TDSam(D, r, D_kp, n):
    """
    核心算法：对时间序列数据进行采样，提取关键点。

    Parameters:
    D (np.ndarray): 输入数据数组
    r (float): 阈值，用于确定关键点
    D_kp (np.ndarray): 输出关键点数组
    n (int): 数据长度
    """
    for i in range(n):
        if i == 0:
            D_kp[0] = D[0]
        elif i == n - 1:
            D_kp[n - 1] = D[n - 1]
        elif 1 <= i < n - 1:
            if D[i - 1] < D[i] > D[i + 1] or D[i - 1] > D[i] < D[i + 1]:
                if D[i] > D[i + 1] + r and D[i] > D[i - 1] + r:
                    D_kp[i] = D[i]
                elif D[i] < D[i + 1] - r and D[i] < D[i - 1] - r:
                    D_kp[i] = D[i]
            elif abs(D[i + 1] - 2 * D[i] + D[i - 1]) > r:
                D_kp[i] = D[i]
            else:
                D_kp[i] = 0  # 不符合条件，填充为 0（可以视为无效点）


def process_csv_files(folder_path, r):

    # 找到所有 CSV 文件
    all_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

    # 读取所有文件并合并
    columns_to_read = ['timestamp', 'P-TPT']
    df_list = [pd.read_csv(file, usecols=columns_to_read) for file in all_files]
    df_combined = pd.concat(df_list, ignore_index=True)

    # 提取 P-TPT 和时间戳
    D = df_combined['P-TPT'].values
    timestamps = df_combined['timestamp'].values
    n = len(D)
    D_kp = np.zeros_like(D)

    # 执行算法
    start_time = time.time()
    TDSam(D, r, D_kp, n)
    end_time = time.time()
    execution_time = end_time - start_time

    result_df = pd.DataFrame({
        'timestamp': timestamps[D_kp != 0],
        'P-TPT': D_kp[D_kp != 0]
    })

    # 输出采样前后数据条数
    print(f"原始数据条数: {len(df_combined)}")
    print(f"采样后数据条数: {len(result_df)}")

    return result_df, execution_time


# 调用封装好的函数
if __name__ == "__main__":
    folder_path = './0'  # 替换为你的 CSV 文件夹路径
    r = 30  # 阈值
    try:
        result_df, exec_time = process_csv_files(folder_path, r)
        print(f"TDSam 算法执行时间: {exec_time:.6f} 秒")
        # print(result_df.head())  # 显示结果的前几行
    except Exception as e:
        print(f"Error: {e}")
