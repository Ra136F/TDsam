import argparse
import http.client
import json
import time

import numpy as np
from twisted.words.protocols.jabber.jstrports import client

from mqt import XenderMQTTClient
from sampler import TDSampler
from util import data_loading

def send2server(data,length,host,port, is_adjust,endpoint="/"):

    try:
        data_json = data.to_dict(orient='records')
        payload = {
            "metadata": {
                "length": length,
                "is_adjust": is_adjust
            },
            "data": data_json  # 数据本体
        }
        body = json.dumps(payload).encode('utf-8')
        # 配置HTTP请求
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body))
        }
        conn = http.client.HTTPConnection(host, port, timeout=600)
        conn.request("POST", endpoint, body=body, headers=headers)
        response = conn.getresponse()
        # print(f"状态码: {response.status}")
        # print(f"响应内容: {response.read().decode()}")
    except Exception as e:
        print(f"请求失败: {e}")
    finally:
        conn.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-data_name', type=str, default='energy', help="数据集名称")
    parser.add_argument('-target', type=str, default='T1', help="目标特征")
    parser.add_argument('-mode', type=int, default=0, help="[0,1],不适用GPU、使用GPU")
    parser.add_argument('-url', type=str, default='http://10.12.54.122:5001/', help="服务器地址")
    args = parser.parse_args()

    folder_path = './data' + '/' + args.data_name
    data, r_min, r_max = data_loading(folder_path, args.target)
    print(f'max{r_max},min:{r_min}')
    sampler=TDSampler(initial_lambda=0.025)
    # result_iloc=sampler.find_key_points(data[args.target].values)
    # result_data = data.iloc[result_iloc].reset_index(drop=True)
    total_rows = len(data)
    batch_rows = int(0.05 * total_rows)
    count=0
    client=XenderMQTTClient(broker="10.12.54.122")
    client.subscribe("xender/control")
    client.client.loop_start()
    is_adjust=False
    start_time=time.time()
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[args.target].values)
        # all_indices = np.arange(len(batch_data))
        # # 排除 result_iloc 的索引（确保转换为集合提高效率）
        # excluded_indices = np.setdiff1d(all_indices, result_iloc)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count+1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        send2server(result_data,len(batch_data)-len(result_data),"10.12.54.122","5001",is_adjust)

        print(f"messages:{client.received_messages}")
        if client.received_messages==1:
            print("调整采样率")
            sampler.lambda_val=sampler.lambda_val//2
            client.received_messages=0
            is_adjust = True
        else:
            is_adjust = False

        print(f"采样率{sampler.lambda_val}")
        count += 1
    # result_iloc = sampler.find_key_points(data[args.target].values)
    # result_data = data.iloc[result_iloc].reset_index(drop=True)
    # send2server(result_data,len(data),"10.12.54.122","5001",is_adjust)
    end_time = time.time()

    print(f"传输完成,共花费:{(end_time-start_time)  : .4f} s")
