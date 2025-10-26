import glob
import http
import json
import math
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
    folder_path = folder_path+".csv"
    # 读取所有文件并合并
    # df_list = [pd.read_csv(file).fillna(0) for file in all_files]
    # df_combined = pd.concat(df_list, ignore_index=True)
    df_combined=pd.read_csv(folder_path)

    D = df_combined[target].values
    min,max = calculate_gap(D)

    return df_combined,min,max

def data_loading_aper(folder_path):
    all_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
    # 读取所有文件并合并
    # dfs = [pd.read_csv(file, sep=';', header=None, names=['date', 'avg_1s', 'avg_8s']) for file in all_files]
    # df_combined = pd.concat(dfs, ignore_index=True)
    df_list = [pd.read_csv(file).fillna(0) for file in all_files]
    df_combined = pd.concat(df_list, ignore_index=True)

    D = df_combined['T-JUS-CKP'].values
    min, max = calculate_gap(D)

    return df_combined, min, max

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


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, np.min(data, 0), np.max(data, 0)

def MinMaxScaler2(data,min,max):
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, min, max

def getMinMax(data,target):
    data=data.drop(columns=['date', 'timestamp'], errors='ignore')
    cols = list(data.columns)
    cols.remove(target)
    data = data[cols+[target]]
    data=data[:int(0.1 * len(data))]
    data, min, max = MinMaxScaler(np.array(data[target]))
    return min, max


class ServerSender:
    def __init__(self, host, port, timeout=600):
        self.conn = http.client.HTTPConnection(host, port, timeout=timeout)

    def send(self, payload, endpoint="/upload"):
        try:
            body = json.dumps(payload).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Content-Length": str(len(body))
            }
            self.conn.request("POST", endpoint, body, headers)
            # response = self.conn.getresponse()
            # response_body = response.read().decode("utf-8")
            response =  self.conn.getresponse()
            status = response.status
            message = None
            response_body = None
            # 1. 尝试解析JSON响应中的消息
            try:
                response_body = response.read().decode('utf-8')
                response_json = json.loads(response_body)
                message = response_json.get('message')
            except:
                # 2. 如果JSON解析失败，直接返回响应体作为消息
                message = response_body if message is None else response_body

            # 3. 如果没有消息内容，默认使用HTTP状态描述
            if message is None:
                message = response.reason
            return status, message
        except Exception as e:
            print(f"[Sender] 请求失败: {e}")
            return 500, str(e)

    def close(self):
        self.conn.close()

def send2server(host,port,conn,send_json,endpoint="/"):
    try:
        payload = send_json
        body = json.dumps(payload).encode('utf-8')
        # 配置HTTP请求
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body))
        }

        conn.request("POST", endpoint, body=body, headers=headers)
        response = conn.getresponse()
        status = response.status
        message = None
        response_body = None
        # 1. 尝试解析JSON响应中的消息
        try:
            response_body = response.read().decode('utf-8')
            response_json = json.loads(response_body)
            message = response_json.get('message')
        except:
            # 2. 如果JSON解析失败，直接返回响应体作为消息
            message = response_body if message is None else response_body

        # 3. 如果没有消息内容，默认使用HTTP状态描述
        if message is None:
            message = response.reason
        return status,message
        # print(f"状态码: {response.status}")
        # print(f"响应内容: {response.read().decode()}")
    except Exception as e:
        print(f"请求失败: {e}")
        return 500, str(e)



def merge_power_data(input_dir, output_file):
    """
    将Tracebase数据集目录中的所有文件合并到单个CSV文件

    参数:
        input_dir: 包含Tracebase数据文件的目录路径
        output_file: 输出的CSV文件路径
    """
    all_data = []

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        # 只处理CSV文件
        if not os.path.isfile(filepath) or not filename.endswith('.csv'):
            continue

        try:
            # 读取CSV文件，使用分号分隔符，跳过空格
            df = pd.read_csv(
                filepath,
                sep=';',
                header=None,
                names=['timestamp_str', 'power_1s', 'power_8s'],
                skipinitialspace=True,  # 忽略值前的空格
                on_bad_lines='skip'  # 跳过格式错误的行
            )

            # 添加处理后的数据到列表
            all_data.append(df)
            print(f"已处理: {filename} ({len(df)} 条记录)")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

    if not all_data:
        print("目录中没有找到有效的CSV文件")
        return

    # 合并所有数据
    combined = pd.concat(all_data, ignore_index=True)

    # 转换时间戳格式
    combined['timestamp'] = pd.to_datetime(
        combined['timestamp_str'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )

    # 删除原始时间戳字符串列
    combined.drop(columns=['timestamp_str'], inplace=True)

    # 删除无效的时间戳
    original_count = len(combined)
    combined = combined.dropna(subset=['timestamp'])
    if len(combined) < original_count:
        removed_count = original_count - len(combined)
        print(f"移除了 {removed_count} 条无效时间戳记录")

    # 按时间戳排序
    combined.sort_values(by='timestamp', inplace=True)

    # 重置索引
    combined.reset_index(drop=True, inplace=True)
    combined.to_csv(output_file, index=False, columns=['timestamp', 'power_1s', 'power_8s'])
    print(f"成功保存 {len(combined)} 条记录到 {output_file}")
    D = combined['power_1s'].values
    min, max = calculate_gap(D)
    return combined,min,max
    # # 保存到CSV



def init_args(config):

    if config.data_name == 'energy':
        config.target = 'T1'
        config.lambda_value = 0.1
        config.second_lambda = 0.01
        config.start_ori_time=40
    elif config.data_name == 'oil-well-1':
        config.target = 'T-JUS-CKP'
        config.lambda_value = 0.1
        if config.method == "c":
            config.k = 100
            config.segment_length = 200
    elif config.data_name == 'household':
        config.lambda_value = 1
        config.second_lambda = 0.5
        config.target = 'Voltage'
        if config.method == "c":
            config.k = 10
            config.segment_length = 200
    elif config.data_name == 'ocean':
        config.target = 'TEMP_2'
        config.lambda_value = 0.01
        config.second_lambda = 0.002
        if config.method == "c":
            config.k = 10
            config.segment_length = 200
    elif config.data_name == 'ppg':
        config.target = 'S1_heart_rate_bpm'
        config.lambda_value = 0.01
        config.second_lambda = 0
        if config.method == "c":
            config.k = 30
            config.segment_length = 200
    elif config.data_name == 'City-power':
        config.target = 'Humidity'
        config.lambda_value = 1.5
        if config.method == "c":
            config.k = 50
            config.segment_length = 250
    elif config.data_name == 'twitter':
        config.target = 'latitude'
        config.lambda_value =15
        if config.method == "c":
            config.k = 5
            config.segment_length = 200
    elif config.data_name == 'rain':
        config.target = 'value'
        config.lambda_value = 1.6
        config.second_lambda=0.2
        config.start_ori_time = 19000
        if config.method == "c":
            config.k = 10
            config.segment_length = 200
    return config

def clean_floats(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # 或者 0.0，看你需求
        return obj
    elif isinstance(obj, dict):
        return {k: clean_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_floats(v) for v in obj]
    else:
        return obj
