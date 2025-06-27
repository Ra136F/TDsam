import time

from sampler import TDSampler
from util import data_loading, getMinMax


def test(config):
    sampler = TDSampler(initial_lambda=0.01, gpu=config.mode)
    folder_path = './data' + '/' + config.data_name
    data, r_min, r_max = data_loading(folder_path, config.target)
    min, max = getMinMax(data, config.target)
    print(f'max{r_max},min:{r_min}')
    sampler = TDSampler(initial_lambda=config.lambda_value)
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


