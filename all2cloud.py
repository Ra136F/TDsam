from mqt import XenderMQTTClient
from sampler import TDSampler
from util import data_loading, send2server


def all_send(config):
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    sampler = TDSampler(initial_lambda=config.lambda_value)
    total_rows = len(data)
    batch_rows = int(config.ratio * total_rows)
    count = 0
    client = XenderMQTTClient(broker=config.ip)
    client.subscribe("xender/control")
    client.client.loop_start()
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        # result_iloc = sampler.find_key_points(batch_data[config.target].values)
        # result_data = batch_data.iloc[result_iloc].reset_index(drop=True)
        print(f"第{count + 1}次传输,原始长度{len(batch_data)}")
        payload = {
            "metadata": {
                "length": len(batch_data),
                "data_name": config.data_name,
                "target": config.target,
            },
            "data": batch_data.to_dict(orient='records'),
        }
        send2server("10.12.54.122", "5002", payload)
        print(f"messages:{client.received_messages}")
        count += 1
