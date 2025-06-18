import argparse
import http.client
import json
import time

import numpy as np
import pandas as pd
from twisted.words.protocols.jabber.jstrports import client

from mqt import XenderMQTTClient
from sampler import TDSampler
from util import data_loading, getMinMax, send2server


def xender_send(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value)
    total_rows = len(data)
    batch_rows = int(config.ratio * total_rows)
    count = 0
    client = XenderMQTTClient(broker="10.12.54.122")
    client.subscribe("xender/control")
    client.client.loop_start()
    is_adjust = False
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        if count % 2 == 0 and count >4:
            ori_data = batch_data
        else:
            ori_data = pd.DataFrame()
        if count != 0:
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": is_adjust,
                    "data_name": config.data_name,
                    "target": config.target
                },
                "data": result_data.to_dict(orient='records'),
                "ori": ori_data.to_dict(orient='records')
            }
        else:
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": is_adjust,
                    "data_name": config.data_name,
                    "target": config.target
                },
                "data": result_data.to_dict(orient='records'),
                "ori": ori_data.to_dict(orient='records'),
                "min": json.dumps(min.tolist()),
                "max": json.dumps(max.tolist())
            }
        send2server("10.12.54.122", "5002", payload)
        print(f"messages:{client.received_messages}")
        if client.received_messages == 1 and sampler.lambda_val!=0:
            print("调整采样率")
            sampler.lambda_val = 0
            client.received_messages = 0
            is_adjust = True
        else:
            is_adjust = False
        print(f"采样率{sampler.lambda_val}")
        count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-data_name', type=str, default='energy', help="数据集名称")
    parser.add_argument('-target', type=str, default='T1', help="目标特征")
    parser.add_argument('-lambda', type=int, default=0.025, help="采样率")
    parser.add_argument('-mode', type=int, default=0, help="[0,1],不适用GPU、使用GPU")
    parser.add_argument('-url', type=str, default='http://10.12.54.122:5002/', help="服务器地址")
    args = parser.parse_args()
    folder_path = './data' + '/' + args.data_name
    data, r_min, r_max = data_loading(folder_path, args.target)
    min,max=getMinMax(data,args.target)
    print(f'max{r_max},min:{r_min}')
    sampler=TDSampler(initial_lambda=0.025)
    total_rows = len(data)
    batch_rows = int(0.05* total_rows)
    count=0
    client=XenderMQTTClient(broker="10.12.54.122")
    client.subscribe("xender/control")
    client.client.loop_start()
    is_adjust=False
    start_time=time.time()
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[args.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")

        if count%2==0:
            ori_data=batch_data
        else:
            ori_data=pd.DataFrame()

        if count!=0:
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": is_adjust
                },
                "data": result_data.to_dict(orient='records'),
                "ori": ori_data.to_dict(orient='records')
            }
        else:
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": is_adjust
                },
                "data": result_data.to_dict(orient='records'),
                "ori": ori_data.to_dict(orient='records'),
                "min": json.dumps(min.tolist()),
                "max": json.dumps(max.tolist())
            }
        send2server("10.12.54.122", "5002",payload)
        print(f"messages:{client.received_messages}")
        if client.received_messages==1:
            print("调整采样率")
            sampler.lambda_val=0
            client.received_messages=0
            is_adjust = True
        else:
            is_adjust = False
        print(f"采样率{sampler.lambda_val}")
        count += 1
    end_time = time.time()
    print(f"传输完成,共花费:{(end_time-start_time): .4f} s")
