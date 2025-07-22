import argparse
import http.client
import json
import time

import joblib
import numpy as np
import pandas as pd


from mqt import XenderMQTTClient
from sampler import TDSampler
from util import data_loading, getMinMax, send2server


def sim_send(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min,max=getMinMax(data,config.target)
    print(f'max{r_max},min:{r_min}')
    sampler=TDSampler(initial_lambda=config.lambda_value)
    total_rows = len(data)
    batch_rows = int(config.ratio* total_rows)
    count=0
    client=XenderMQTTClient(broker=config.ip)
    client.subscribe("xender/control")
    client.client.loop_start()
    # batch_rows=160+128
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        result_iloc = sampler.find_key_points(batch_data[config.target].values)
        result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        # result_data = batch_data[:160]
        if config.data_name=="oil-well":
            result_data = batch_data[:int(len(batch_data)*0.5)]
        print(f"第{count + 1}次采样,原始长度{len(batch_data)},采样长度:{len(result_data)}")
        payload = {
            "metadata": {
                "length": len(batch_data)-len(result_data),
                "data_name": config.data_name,
                "target": config.target,
            },
            "data": result_data.to_dict(orient='records'),
        }
        send2server("10.12.54.122", "5002",payload)
        print(f"messages:{client.received_messages}")
        print(f"采样率{sampler.lambda_val}")
        count += 1

