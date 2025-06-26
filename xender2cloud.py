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
    # client.subscribe("xender/control")
    # client.client.loop_start()
    is_adjust = False
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        if  count%2==0 and  2 < count < 60 and sampler.lambda_val == config.lambda_value:
            ori_data = batch_data
            # ori_data = pd.DataFrame()
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
        status,message = send2server("10.12.54.122", "5002", payload)
        if status == 200:
            print(f"Server response: message={message}")
            if message == 1 and sampler.lambda_val == config.lambda_value:
                print("调整采样率")
                sampler.lambda_val = 0.05
                is_adjust = True
            else:
                is_adjust = False
            # if client.received_messages == 1 and sampler.lambda_val == config.lambda_value:
            #     print("调整采样率")
            #     sampler.lambda_val = 0
            #     client.received_messages = 0
            #     is_adjust = True
            # else:
            #     is_adjust = False
        else:
            print(f"请求失败，状态码: {status}")
            break
        print(f"采样率{sampler.lambda_val}")
        count += 1



def fenlei_send(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value)
    total_rows = len(data)
    batch_rows = int(config.ratio * total_rows)
    count = 0
    is_adjust = False
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        if count != 0:
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": is_adjust,
                    "data_name": config.data_name,
                    "target": config.target
                },
                "data": result_data.to_dict(orient='records')
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
                "min": json.dumps(min.tolist()),
                "max": json.dumps(max.tolist())
            }
        status,message =send2server("10.12.54.122", "5002", payload)
        if status==200:
            print(f"采样率{sampler.lambda_val}")
        count += 1
