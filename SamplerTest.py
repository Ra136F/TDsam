import json
import time

import numpy as np

from cusum import AdaptiveCUSUM
from sampler import TDSampler
from util import data_loading, getMinMax, send2server, MinMaxScaler2, MinMaxScaler

from joblib import  load


def test(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min, max = getMinMax(data, config.target)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value,gpu=config.mode)
    total_rows = len(data)
    batch_rows = int(config.ratio * total_rows)
    count = 0
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        start = time.time()
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        end = time.time()
        run_time = end - start
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)},运行时间:{run_time}s")


def local_fenlei_cusum(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min, max = getMinMax(data, config.target)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value, gpu=config.mode)
    count = 0
    is_adjust = False
    detector = AdaptiveCUSUM(k=100, drift_k=0.5, min_sigma=0.1, alpha=0.1, min_segment_length=200)
    detected_change_points = []
    last_cp = 0
    is_last = False
    cols=list(data.columns)
    if "date" in cols:
        cols.remove("date")
    elif "timestamp" in cols:
        cols.remove("timestamp")
    cols.remove(config.target)
    cols=cols+[config.target]
    print(cols)
    kmmodel = load(f'./model/{config.data_name}-km-200.pkl')['model']
    for i, (_, row) in enumerate(data.iterrows()):
        value = row[config.target]
        change_detected, position = detector.update(value)
        if change_detected:
            detected_change_points.append(i)
            # 发送上一个变点到当前变点之间的数据
            if last_cp < i:  # 确保有数据可发送
                batch_data = data[last_cp:i]
                fenlei_data = batch_data[cols].values
                fenlei_data, _, _ = MinMaxScaler2(fenlei_data, min, max)
                fenlei_data = fenlei_data[:, -1]
                fenlei_data = fenlei_data.reshape(1, len(fenlei_data), 1)
                labels = kmmodel.predict(fenlei_data)
                model_id = labels[0]
                print(f"选择模型:{model_id}")
                result_iloc = sampler.find_key_points(batch_data[config.target].values)
                result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
                # fenlei_data=result_data[cols].values
                # fenlei_data,_,_=MinMaxScaler2(fenlei_data,min,max)
                # fenlei_data = fenlei_data[:,-1]
                # fenlei_data = fenlei_data.reshape(1, len(fenlei_data), 1)
                # labels = kmmodel.predict(fenlei_data)
                # model_id = labels[0]
                # print(f"选择模型:{model_id}")
                print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
                if count != 0:
                    payload = {
                        "metadata": {
                            "length": len(batch_data),
                            "is_adjust": is_adjust,
                            "data_name": config.data_name,
                            "target": config.target,
                            "is_last": is_last,  # 添加结束标记
                            "model_id": int(model_id)
                        },
                        "data": result_data.to_dict(orient='records'),
                    }
                else:
                    payload = {
                        "metadata": {
                            "length": len(batch_data),
                            "is_adjust": is_adjust,
                            "data_name": config.data_name,
                            "target": config.target,
                            "is_last": is_last,  # 添加结束标记
                            "model_id": int(model_id)
                        },
                        "data": result_data.to_dict(orient='records'),
                        "min": json.dumps(min.tolist()),
                        "max": json.dumps(max.tolist()),

                    }
                status, message = send2server("10.12.54.122", "5002", payload)
                if status == 200:
                    print(f"采样率{sampler.lambda_val}")
                count += 1
            last_cp = i
    if last_cp < len(data):
        is_last = True
        batch_data = data[last_cp:]
        fenlei_data = batch_data[cols].values
        fenlei_data, _, _ = MinMaxScaler2(fenlei_data, min, max)
        fenlei_data = fenlei_data[:, -1]
        fenlei_data = fenlei_data.reshape(1, len(fenlei_data), 1)
        labels = kmmodel.predict(fenlei_data)
        model_id = labels[0]
        print(f"选择模型:{model_id}")
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        # fenlei_data = result_data[cols].values
        # fenlei_data, _, _ = MinMaxScaler2(fenlei_data, min, max)
        # fenlei_data = fenlei_data[:, -1]
        # fenlei_data = fenlei_data.reshape(1, len(fenlei_data), 1)
        # labels = kmmodel.predict(fenlei_data)
        # model_id = labels[0]
        # print(f"选择模型:{model_id}")
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        payload = {
            "metadata": {
                "length": len(batch_data),
                "is_adjust": is_adjust,
                "data_name": config.data_name,
                "target": config.target,
                "is_last": is_last , # 添加结束标记
                "model_id": int(model_id)
            },
            "data": result_data.to_dict(orient='records'),

        }
        status, message = send2server("10.12.54.122", "5002", payload)
        if status == 200:
            print(f"采样率{sampler.lambda_val}")
        print("传输完成")
        count += 1



def local_fenlei_guding(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min, max = getMinMax(data, config.target)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value, gpu=config.mode)
    total_rows = len(data)
    batch_rows = config.group
    if config.group == 0:
        batch_rows = int(config.ratio * total_rows)
    count = 0
    is_adjust = False
    total_batches = total_rows // batch_rows
    if total_rows % batch_rows != 0:
        total_batches += 1
    print(f"总批次:{total_batches}")
    kmmodel = load(f'./model/{config.data_name}-km-o.pkl')['model']
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        fenlei_data = batch_data[config.target].values
        fenlei_data, _, _ = MinMaxScaler(fenlei_data)
        fenlei_data = fenlei_data.reshape(1, len(fenlei_data), 1)
        if len(batch_data) < config.group:
            model_id = 0
        else:
            labels = kmmodel.predict(fenlei_data)
            model_id = labels[0]
        print(f"选择模型:{model_id}")
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        # fenlei_data = result_data[config.target].values
        # fenlei_data, _, _ = MinMaxScaler(fenlei_data)
        # fenlei_data = fenlei_data.reshape(1, len(fenlei_data), 1)
        # labels = kmmodel.predict(fenlei_data)
        # model_id = labels[0]
        # print(f"选择模型:{model_id}")
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        is_last = (i + batch_rows >= total_rows)
        if count != 0:
            payload = {
                "metadata": {
                    "length": len(batch_data),
                    "is_adjust": is_adjust,
                    "data_name": config.data_name,
                    "target": config.target,
                    "is_last": is_last,  # 添加结束标记
                    "model_id": int(model_id)
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
                    "is_last": is_last,  # 添加结束标记
                    "model_id": int(model_id)
                },
                "data": result_data.to_dict(orient='records'),
                "min": json.dumps(min.tolist()),
                "max": json.dumps(max.tolist())
            }
        status, message = send2server("10.12.54.122", "5002", payload)
        if status == 200:
            print(f"采样率{sampler.lambda_val}")
        count += 1

