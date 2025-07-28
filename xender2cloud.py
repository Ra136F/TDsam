import argparse
import http.client
import json
import time

import numpy as np
import pandas as pd

import asyncio
from concurrent.futures import ThreadPoolExecutor
from cusum import AdaptiveCUSUM
from mqt import XenderMQTTClient
from sampler import TDSampler
from util import data_loading, getMinMax, send2server


def xender_send(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value,gpu=config.mode)
    total_rows = len(data)
    batch_rows = config.group
    if config.group == 0:
        batch_rows = int(config.ratio * total_rows)
    count = 0
    client = XenderMQTTClient(broker="10.12.54.122")
    # client.subscribe("xender/control")
    # client.client.loop_start()
    is_adjust = False
    total_batches = total_rows // batch_rows
    if total_rows % batch_rows != 0:
        total_batches += 1
    print(f"总批次:{total_batches}")
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        is_last = (i + batch_rows >= total_rows)
        if   250 < count  and sampler.lambda_val == config.lambda_value:
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
                    "target": config.target,
                    "is_last": is_last

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
                    "target": config.target,
                    "is_last": is_last
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
                sampler.lambda_val =0.001
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






#固定分组
def fenlei_send(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value,gpu=config.mode)
    total_rows = len(data)
    batch_rows = config.group
    if config.group ==0:
        batch_rows = int(config.ratio * total_rows)
    count = 0
    is_adjust = False
    total_batches = total_rows // batch_rows
    if total_rows % batch_rows != 0:
        total_batches += 1
    print(f"总批次:{total_batches}")
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        is_last = (i + batch_rows >= total_rows)
        if count != 0:
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": is_adjust,
                    "data_name": config.data_name,
                    "target": config.target,
                    "is_last": is_last  # 添加结束标记
                },
                "data": result_data.to_dict(orient='records')
            }
        else:
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": is_adjust,
                    "data_name": config.data_name,
                    "target": config.target,
                    "is_last": is_last  # 添加结束标记
                },
                "data": result_data.to_dict(orient='records'),
                "min": json.dumps(min.tolist()),
                "max": json.dumps(max.tolist())
            }
        status,message =send2server("10.12.54.122", "5002", payload)
        if status==200:
            print(f"采样率{sampler.lambda_val}")
        count += 1



#在线检测
def fenlei_send2(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(min)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value,gpu=config.mode)
    count = 0
    is_adjust = False
    detector = AdaptiveCUSUM(k=10, drift_k=0.5, min_sigma=0.1, alpha=0.1, min_segment_length=200)
    detected_change_points = []
    last_cp = 0
    is_last=False
    for i, (_, row) in enumerate(data.iterrows()):
        value = row[config.target]
        change_detected, position = detector.update(value)
        if change_detected:
            detected_change_points.append(i)
            # 发送上一个变点到当前变点之间的数据
            if last_cp < i:  # 确保有数据可发送
                batch_data=data[last_cp:i]
                result_iloc = sampler.find_key_points(batch_data[config.target].values)
                result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
                print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
                if count != 0:
                    payload = {
                        "metadata": {
                            "length": len(batch_data),
                            "is_adjust": is_adjust,
                            "data_name": config.data_name,
                            "target": config.target,
                            "is_last": is_last  # 添加结束标记
                        },
                        "data": result_data.to_dict(orient='records')
                    }
                else:
                    payload = {
                        "metadata": {
                            "length": len(batch_data),
                            "is_adjust": is_adjust,
                            "data_name": config.data_name,
                            "target": config.target,
                            "is_last": is_last  # 添加结束标记
                        },
                        "data": result_data.to_dict(orient='records'),
                        "min": json.dumps(min.tolist()),
                        "max": json.dumps(max.tolist())
                    }
                status, message = send2server("10.12.54.122", "5002", payload)
                if status==200:
                    print(f"采样率{sampler.lambda_val}")
                count+=1
            last_cp = i
    if last_cp < len(data):
        is_last=True
        batch_data = data[last_cp:]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        payload = {
            "metadata": {
                "length": len(batch_data),
                "is_adjust": is_adjust,
                "data_name": config.data_name,
                "target": config.target,
                "is_last": is_last  # 添加结束标记
            },
            "data": result_data.to_dict(orient='records')
        }
        status, message = send2server("10.12.54.122", "5002", payload)
        if status==200:
            print(f"采样率{sampler.lambda_val}")
        print("传输完成")
        count+=1
    # print(f"检测到的突变点位置: {detected_change_points}")

