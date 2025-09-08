import argparse
import http.client
import json
import time
import threading
import queue

import httpx
import numpy as np
import pandas as pd

import asyncio
from concurrent.futures import ThreadPoolExecutor
from cusum import AdaptiveCUSUM
from mqt import XenderMQTTClient
from numbacusum import NumbaCUSUM
from sampler import TDSampler, RandomSampler, RandomSampler2
from util import data_loading, getMinMax, send2server, clean_floats, ServerSender





def xender_send_async(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min_val, max_val = getMinMax(data, config.target)

    sampler = TDSampler(initial_lambda=config.lambda_value, gpu=config.mode)
    total_rows = len(data)
    batch_rows = config.group if config.group > 0 else int(config.ratio * total_rows)
    total_batches = (total_rows + batch_rows - 1) // batch_rows
    print(f"总批次:{total_batches}")
    mqtt_client = XenderMQTTClient(broker="10.12.54.122")
    batch_data = data[:batch_rows]
    result_iloc = sampler.find_key_points(batch_data[config.target].values)
    result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
    ori_data = pd.DataFrame()  # 第一批不传 ori
    is_last = (batch_rows >= total_rows)
    is_adjust=False
    print(is_last)
    payload = {
        "metadata": {
            "batch_id": 0,
            "length": len(batch_data),
            "is_adjust": False,
            "data_name": config.data_name,
            "target": config.target,
            "is_last": is_last,
        },
        "data": result_data.to_dict(orient='records'),
        "ori": ori_data.to_dict(orient='records'),
        "min": json.dumps(min_val.tolist()),
        "max": json.dumps(max_val.tolist())
    }
    print("发送第一批数据（同步）...")
    status, message = send2server("10.12.54.122", "5002", payload)
    if status != 200:
        print(f"请求失败，状态码: {status}")
        return
    print("第一批处理完成，服务端已初始化")
    count = 1
    for i in range(batch_rows, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        is_last = (i + batch_rows >= total_rows)
        ori_data = batch_data if (config.aware == 1 and count >= config.start_ori_time and sampler.lambda_val==config.lambda_value ) else pd.DataFrame()
        print(f"是否传输原始数据:{len(ori_data)>0}")
        payload = {
            "metadata": {
                "batch_id": count,
                "length": len(batch_data),
                "is_adjust": is_adjust,
                "data_name": config.data_name,
                "target": config.target,
                "is_last": is_last
            },
            "data": result_data.to_dict(orient='records'),
            "ori": ori_data.to_dict(orient='records')
        }
        status, message = send2server("10.12.54.122", "5002", payload)
        if status != 200:
            print(f"请求失败，状态码: {status}")
            break
        print(f"第{count + 1}批发送完成，采样率={sampler.lambda_val}")
        # 检查是否收到采样率调整信号
        mqtt_client.client.loop(timeout=0)
        if mqtt_client.received_adjust:
            print("[Client] 采样率调整信号生效")
            sampler.lambda_val = config.second_lambda
            is_adjust = True
            mqtt_client.received_adjust = False
        else:
            is_adjust = False
        count += 1

    # === Step 3: 等待完成通知 ===
    while not mqtt_client.received_done:
        time.sleep(0.1)
    print("客户端确认：所有批次处理并保存完成")

def xender_pipeline_send2(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min_val, max_val = getMinMax(data, config.target)

    sampler = TDSampler(initial_lambda=config.lambda_value, gpu=config.mode)
    total_rows = len(data)
    batch_rows = config.group or int(config.ratio * total_rows)
    total_batches = (total_rows + batch_rows - 1) // batch_rows
    print(f"总批次: {total_batches}")

    # 队列：采样线程 -> 传输线程
    q = queue.Queue(maxsize=5)

    # 事件标志：传输线程 -> 采样线程
    adjust_event = threading.Event()

    # -------- 采样线程 --------
    def sampler_worker():
        count = 0
        for i in range(0, total_rows, batch_rows):
            batch_data = data[i:i + batch_rows]
            result_iloc = sampler.find_key_points(batch_data[config.target].values)
            result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": False,  # 默认 False
                    "data_name": config.data_name,
                    "target": config.target,
                    "is_last": (i + batch_rows >= total_rows),
                },
                "data": result_data.to_dict(orient='records'),
                "ori": batch_data.to_dict(orient='records') if (config.aware and count >=config.start_ori_time) else [],
            }
            # 第一批加 min/max
            if count == 0:
                payload["min"] = json.dumps(min_val.tolist())
                payload["max"] = json.dumps(max_val.tolist())

            # 检查是否需要触发模型更新
            if adjust_event.is_set():
                payload["metadata"]["is_adjust"] = True
                adjust_event.clear()  # 用掉一次就清空，避免重复
                print(f"[Sampler] 第{count+1}批将触发模型更新 (is_adjust=True)")

            q.put(payload)
            print(f"[Sampler] 已采样第{count+1}批, 原始={len(batch_data)}, 采样={len(result_data)}")
            count += 1

        q.put(None)  # 结束信号

    sender = ServerSender("10.12.54.122", 5002)
    # -------- 传输线程 --------
    def sender_worker():
        nonlocal sampler
        count = 0
        while True:
            payload = q.get()
            try:
                if payload is None:
                    break
                # status, message = send2server("10.12.54.122", "5002", payload, endpoint="/upload")
                status, message = sender.send(payload, "/upload")
                count += 1
                print(f"[Sender] 第{count}批响应: {status}, {message},采样率,{sampler.lambda_val}")

                # -------- 采样率调整逻辑 --------
                if status == 200 and message == 1 and sampler.lambda_val == config.lambda_value:
                    print("[Sender] 服务端要求调整采样率")
                    # 调整采样率
                    sampler.lambda_val = config.second_lambda
                    print(f"[Sender] 采样率已从 {config.lambda_value} 调整为 {sampler.lambda_val}")
                    # 通知采样线程，下一批 is_adjust=True
                    adjust_event.set()
            except Exception as e:
                print(f"[Sender] 发送异常: {e}")
            finally:
                q.task_done()  #
    # 启动线程
    t1 = threading.Thread(target=sampler_worker)
    t2 = threading.Thread(target=sender_worker)
    t1.start()
    t2.start()
    # 等待采样完成
    t1.join()
    # 等待所有传输完成
    q.join()
    # 通知发送线程退出
    q.put(None)
    t2.join()
    print("全部数据采样 + 传输完成")

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
    conn = http.client.HTTPConnection("10.12.54.122", 5002, timeout=600)
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
        if  config.start_ori_time < count  and sampler.lambda_val == config.lambda_value:
            if config.aware==0:
                ori_data = pd.DataFrame()
            else:
                ori_data = batch_data
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
        status,message = send2server("10.12.54.122", "5002",conn, payload)
        if status == 200:
            print(f"Server response: message={message}")
            if message == 1 and sampler.lambda_val == config.lambda_value:
                print("调整采样率")
                sampler.lambda_val =config.second_lambda
                is_adjust = True
            else:
                is_adjust = False
        else:
            print(f"请求失败，状态码: {status}")
            break
        print(f"采样率{sampler.lambda_val}")
        count += 1
    conn.close()

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
    last_lambda = 0
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        if is_adjust:
            result_data = batch_data.copy()
            # result_iloc = sampler.find_key_points(batch_data[config.target].values)
            # result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
            sampler.lambda_val = last_lambda
        else:
            result_iloc = sampler.find_key_points(batch_data[config.target].values)
            result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        is_last = (i + batch_rows >= total_rows)
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
        # 第一批数据添加min/max
        if count == 0:
            payload["min"] = json.dumps(min.tolist())
            payload["max"] = json.dumps(max.tolist())
        status,message =send2server("10.12.54.122", "5002", payload)
        if message == 1:
            last_lambda = sampler.lambda_val
            sampler.lambda_val = -1
            is_adjust = True
        else:
            is_adjust = False
        if status==200:
            print(f"采样率{sampler.lambda_val}")
        count += 1



#在线检测cusum
def fenlei_send2(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(min)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value,gpu=config.mode)
    count = 0
    is_adjust = False
    detector = AdaptiveCUSUM(k=20, drift_k=0.5, min_sigma=0.1, alpha=0.1, min_segment_length=200)
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
                if sampler.lambda_val == -1:
                    result_data=[]
                    sampler.lambda_val =config.lambda_value
                else:
                    result_iloc = sampler.find_key_points(batch_data[config.target].values)
                    result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
                print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
                # if count != 0:
                #     payload = {
                #         "metadata": {
                #             "length": len(batch_data),
                #             "is_adjust": is_adjust,
                #             "data_name": config.data_name,
                #             "target": config.target,
                #             "is_last": is_last  # 添加结束标记
                #         },
                #         "data": result_data.to_dict(orient='records')
                #     }
                # else:
                #     payload = {
                #         "metadata": {
                #             "length": len(batch_data),
                #             "is_adjust": is_adjust,
                #             "data_name": config.data_name,
                #             "target": config.target,
                #             "is_last": is_last  # 添加结束标记
                #         },
                #         "data": result_data.to_dict(orient='records'),
                #         "min": json.dumps(min.tolist()),
                #         "max": json.dumps(max.tolist())
                #     }
                payload = {
                    "metadata": {
                        "length": len(batch_data),
                        "is_adjust": False,
                        "data_name": config.data_name,
                        "target": config.target,
                        "is_last": is_last  # 添加结束标记
                    },
                    "data": result_data.to_dict(orient='records')
                }
                # 第一批数据添加min/max
                if len(detected_change_points) == 1:
                    payload["min"] = json.dumps(min.tolist())
                    payload["max"] = json.dumps(max.tolist())
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




#先找点再发送
def fenlei_send3(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(min)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value,gpu=config.mode)
    count = 0
    is_adjust = False
    last_cp = 0
    is_last=False
    start_time = time.time()
    detector = NumbaCUSUM(k=10, drift_k=0.5, min_sigma=0.1, alpha=0.1, min_segment_length=200)
    changes = detector.batch_update(data[config.target].values)
    detected_change_points = [i for i, changed in enumerate(changes) if changed]
    print(len(detected_change_points))
    end_time = time.time()
    print(f"找点花费{end_time - start_time}s")
    for i in detected_change_points:
        batch_data=data[last_cp:i]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        payload = {
            "metadata": {
                "length": len(batch_data),
                "is_adjust": False,
                "data_name": config.data_name,
                "target": config.target,
                "is_last": is_last  # 添加结束标记
            },
            "data": result_data.to_dict(orient='records')
        }
        # 第一批数据添加min/max
        if count == 0:
            payload["min"] = json.dumps(min.tolist())
            payload["max"] = json.dumps(max.tolist())
        status, message = send2server("10.12.54.122", "5002", payload)
        if status == 200:
            print(f"采样率{sampler.lambda_val}")
        count += 1
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

#动态增加类别
def fenlei_send4(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(min)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value, gpu=config.mode)
    if config.sampler=='random':
        sampler=RandomSampler2(sample_prob=0.3)

    count = 0
    is_adjust = False
    detector = AdaptiveCUSUM(k=config.k, drift_k=0.5, min_sigma=0.1, alpha=0.1, min_segment_length=config.segment_length)
    detected_change_points = []
    last_cp = 0
    last_lambda = 0
    is_last=False
    for i, (_, row) in enumerate(data.iterrows()):
        value = row[config.target]
        change_detected, position = detector.update(value)
        if change_detected:
            detected_change_points.append(i)
            # 发送上一个变点到当前变点之间的数据
            if last_cp < i:  # 确保有数据可发送
                batch_data=data[last_cp:i]
                if is_adjust:
                    result_data=batch_data.copy()
                    # result_iloc = sampler.find_key_points(batch_data[config.target].values)
                    # result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
                    sampler.lambda_val =last_lambda
                else:
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
                # 第一批数据添加min/max
                if len(detected_change_points) == 1:
                    payload["min"] = json.dumps(min.tolist())
                    payload["max"] = json.dumps(max.tolist())
                status, message = send2server("10.12.54.122", "5002", payload)
                if message==1:
                    last_lambda = sampler.lambda_val
                    sampler.lambda_val = -1
                    is_adjust=True
                else:
                    is_adjust=False
                if status==200:
                    if config.sampler=='random':
                        print(f"随机采样10%数据")
                    else:
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