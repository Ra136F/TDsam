import http

from mqt import XenderMQTTClient
from sampler import TDSampler
from util import data_loading, send2server


def all_send(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    total_rows = len(data)
    batch_rows = config.group
    if config.group == 0:
        batch_rows = int(config.ratio * total_rows)
    count = 0
    total_batches = total_rows // batch_rows
    conn = http.client.HTTPConnection("10.12.54.122", 5002, timeout=600)
    if total_rows % batch_rows != 0:
        total_batches += 1
    # client = XenderMQTTClient(broker=config.ip)
    # client.subscribe("xender/control")
    # client.client.loop_start()
    print(f"总批次:{total_batches}")
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        print(f"第{count + 1}次传输,原始长度{len(batch_data)}")
        is_last = (i + batch_rows >= total_rows)
        payload = {
            "metadata": {
                "length": len(batch_data),
                "data_name": config.data_name,
                "target": config.target,
                "is_last": is_last,
            },
            "data": batch_data.to_dict(orient='records'),
        }
        status,message =send2server("10.12.54.122", "5002",conn, payload)
        if status == 200:
            print(f"第{count + 1}次传输完成")
        count += 1
