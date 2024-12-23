import os, sys
import platform
import hashlib
import json
import time
import datetime
import string
import multiprocessing
import threading, queue
import psutil
import operator, random
# import settings
import requests
from datetime import datetime
import TDSam2
from TDSam2 import process_csv_files

def send_to_server(result_df, url):
    # 转换 DataFrame 为 JSON 格式
    result_json = result_df.to_json(orient='records')  # 转换为记录格式的 JSON
    # 发送 POST 请求
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=result_json, headers=headers)
        response.raise_for_status()  # 检查 HTTP 请求是否成功
        print(f"数据成功发送到服务器，响应状态码: {response.status_code}")
        return response
    except requests.exceptions.RequestException as e:
        print(f"发送数据时出错: {e}")
        return None



if __name__ == '__main__':
    # 获取当前时间
    current_time = datetime.now()
    # 格式化输出
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间1:", formatted_time)

    folder_path = './0'  # 替换为你的 CSV 文件夹路径
    r = 10  # 阈值
    url='http://10.12.54.122:5001/upload'
    result_df, exec_time = process_csv_files(folder_path, r)
    print(f"TDSam 算法执行时间: {exec_time:.6f} 秒")

    send_to_server(result_df, url)
    # main2()



