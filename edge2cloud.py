import glob
import os, sys
import platform
import hashlib
import json
import time
import datetime
import string
import multiprocessing
import threading, queue

import numpy as np
import pandas as pd
import psutil
import operator, random
# import settings
import requests
from datetime import datetime
import TDSam2
import argparse

def data_loading(folder_path,target):
    all_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
    # 读取所有文件并合并
    df_list = [pd.read_csv(file).fillna(0) for file in all_files]
    df_combined = pd.concat(df_list, ignore_index=True)
    print(f"原始数据条数: {len(df_combined)}")
    D = df_combined[target].values
    r = calculate_delta(D)
    print(f"δ is {r}")
    return df_combined,r

def calculate_delta(data):
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
    delta_value = np.min(non_zero_differences)
    return delta_value


# def process_csv_files(folder_path, target,mode):
#     # 找到所有 CSV 文件
#     all_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
#     if not all_files:
#         raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
#     # 读取所有文件并合并
#     df_list = [pd.read_csv(file).fillna(0) for file in all_files]
#     df_combined = pd.concat(df_list, ignore_index=True)
#     print(f"原始数据条数: {len(df_combined)}")
#     D = df_combined[target].values
#     r= calculate_delta(D)
#     r=r*4
#     print(f"δ is {r}")
#     n = len(D)
#     D_kp = np.zeros_like(D)
#     is_key_point = np.zeros(n, dtype=bool)
#     # 执行算法
#     start_time = time.time()
#     TDSam2.TDSam(D, r, D_kp)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     mask = is_key_point
#     result_df = df_combined[mask].copy()
#     result_df[target] = D_kp[mask]
#     # 输出采样前后数据条数
#     print(f"采样后数据条数: {len(result_df)}")
#     diff_len=len(df_combined)-len(result_df)
#     print(f"减少了: {diff_len}")
#     return result_df, execution_time
# def process_ori(folder_path, target):
#     all_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
#     if not all_files:
#         raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
#     # 读取所有文件并合并
#     df_list = [pd.read_csv(file).fillna(0) for file in all_files]
#     df_combined = pd.concat(df_list, ignore_index=True)
#     print(f"原始数据条数: {len(df_combined)}")
#     return df_combined ,len(df_combined)



def send_to_server(result_df, url,data_name,target):
    # 转换 DataFrame 为 JSON 格式
    result_json = {
        "data_name": data_name,
        "target": target,
        "result_data": result_df.to_dict(orient='records')
    }
    # 发送 POST 请求
    try:
        headers = {'Content-Type': 'application/json'}
        start_time = time.time()
        response = requests.post(url, data=json.dumps(result_json), headers=headers)
        response.raise_for_status()  # 检查 HTTP 请求是否成功
        end_time = time.time()
        transfer_time = end_time - start_time
        print(f"数据成功发送到服务器，响应状态码: {response.status_code}")
        print(f"数据传输完成耗时: {transfer_time:.6f} 秒")
        resp_data=json.loads(response.text)
        return resp_data, transfer_time
    except requests.exceptions.RequestException as e:
        print(f"发送数据时出错: {e}")
        return None

def execute_sample(data,args,r):
    data = np.asarray(data)
    n = len(data)
    start_time = time.time()
    result=TDSam2.TDSam(data, r, n)
    end_time = time.time()
    execution_time = end_time - start_time
    # 输出采样前后数据条数
    print(f"采样后数据条数: {len(result)}")
    diff_len = n - len(result)
    print(f"减少了: {diff_len}")
    result = pd.DataFrame(result, columns=[args.target])
    return result, execution_time


#找到数据集最佳采样点
def find_delta(data, args,r):
    is_done=False
    final_r=r
    url=args.url+'adjust'
    count=1
    while not is_done:
        sample_data,execution_time=execute_sample(data,args,final_r)
        print(f'第{count}次传输开始...,采样率是{final_r}')
        resp_data, transfer_time=send_to_server(sample_data,url,args.data_name,args.target)
        if resp_data.get('isAdjust')==1:
            final_r+=r
        else:
            is_done=True
        count+=1
    return final_r



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-data_name', type=str, default='oil-well', help="数据集名称")
    parser.add_argument('-target', type=str, default='P-TPT', help="目标特征")
    parser.add_argument('-mode', type=int, default=0, help="[0,1],不适用GPU、使用GPU")
    parser.add_argument('-url', type=str, default='http://10.12.54.122:5001/', help="服务器地址")
    args = parser.parse_args()

    # 获取当前时间
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间:", formatted_time)
    folder_path = './data'+'/'+args.data_name+'/0'
    data,r= data_loading(folder_path,args.target) #加载数据,计算δ
    r=find_delta(data[args.target].values,args,r)
    print(f'最终的λ:{r}')
    # main2()



