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

from util import data_loading


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



def send_to_server(result_df, url,data_name,target,gen_len,count,last_dtw=0):
    # 转换 DataFrame 为 JSON 格式
    result_json = {
        "data_name": data_name,
        "target": target,
        "result_data": result_df.to_dict(orient='records'),
        "length": gen_len, #生成数据的长度
        "count": count,
        "last_dtw": last_dtw
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
    return result, execution_time, diff_len


#找到数据集最佳采样点
def find_best(folder_path, args):
    data, s_min, s_max = data_loading(folder_path, args.target)
    # data = data[:int(0.05*len(data))]
    is_done=False
    final_r=0
    url= args.url + 'adjust'
    count=1
    last_dtw=1e10
    while not is_done:
        sample_data,execution_time,diff_len=execute_sample(data, args, final_r)
        print(f'第{count}次传输开始...,采样率是{final_r}')
        resp_data, transfer_time=send_to_server(sample_data, url, args.data_name, args.target,diff_len,count,last_dtw)
        if resp_data.get('isAdjust')==1 and final_r<=s_max:
            final_r+=s_min
            last_dtw=resp_data.get("last_dtw")
        else:
            final_r-=s_min
            is_done=True
        count+=1
    return final_r
#自适应调整
def send_to_server2(folder_path,args,r):
    data, s_min, s_max = data_loading(folder_path, args.target)
    s_min=0.005
    s_max=0.25
    resp_data=0
    count=0
    total_rows=len(data)
    batch_rows=int(0.05*total_rows)
    start_time = time.time()
    for i in range(0,total_rows,batch_rows):
        batch_data=data[i:i+batch_rows]
        sample_data,execution_time,diff_len=execute_sample(batch_data[args.target].values, args, r)
        resp_data, transfer_time=send_to_server(sample_data, args.url+'/upload', args.data_name, args.target,diff_len,count)
        count+=1
    print(count)
    end_time = time.time()
    print(f'分段发送共耗时:{(end_time-start_time):.4f}s')
    start_time = time.time()
    sample_data, execution_time, diff_len = execute_sample(data[args.target].values, args, r)
    resp_data, transfer_time = send_to_server(sample_data, args.url + '/upload', args.data_name, args.target, diff_len,
                                              count)
    end_time = time.time()
    print(f'总发送耗时:{(end_time-start_time):.4f}s')
    return resp_data


def main(args):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间:", formatted_time)
    folder_path = './data' + '/' + args.data_name
    # data, r_min, r_max = data_loading(folder_path, args.target)  # 加载数据,计算δ
    # send_ori(args)
    # r = find_best(folder_path , args)
    # print(f'最终的λ:{r}')
    send_to_server2(folder_path,args,0.05)


#传输原始数据
def send_ori(args):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间:", formatted_time)
    folder_path = './data' + '/' + args.data_name+'/9'
    data, r_min, r_max = data_loading(folder_path, args.target)  # 加载数据,计算δ
    # data= pd.DataFrame(data, columns=[args.target])
    # data = data[:int(0.5*len(data))]
    resp_data, transfer_time=send_to_server(data,args.url+'upload',args.data_name,args.target,0,0,0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-data_name', type=str, default='energy', help="数据集名称")
    parser.add_argument('-target', type=str, default='T1', help="目标特征")
    parser.add_argument('-mode', type=int, default=0, help="[0,1],不适用GPU、使用GPU")
    parser.add_argument('-url', type=str, default='http://10.12.54.122:5001/', help="服务器地址")
    args = parser.parse_args()
    main(args)
    # send_ori(args)
    # 获取当前时间

    # main2()



