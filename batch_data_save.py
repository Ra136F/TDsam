import argparse
import os

from util import data_loading

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-data_name', type=str, default='oil-well-1', help="数据集名称")
    parser.add_argument('-target', type=str, default='T-JUS-CKP', help="目标特征")
    parser.add_argument('-ip', type=str, default='10.12.54.122', help="IP地址")
    parser.add_argument('-port', type=str, default='5001', help="端口")
    parser.add_argument('-ratio', type=float, default=0.002, help="比例")
    parser.add_argument('-group', type=int, default=300, help='分组')
    parser.add_argument('-compressOption', type=str, default='adaptive', help="压缩策略")
    parser.add_argument('-network', type=str, default='wlan0', help='网卡名称')
    parser.add_argument("-upload_bandwidth", type=float, default=0.1, help="上传带宽")
    parser.add_argument("-download_bandwidth", type=float, default=0.1, help="下载带宽")
    args = parser.parse_args()

    data_path = "./data/" + args.data_name
    data_next_path="./data/"+args.data_name+"/"
    os.makedirs(os.path.dirname(data_next_path), exist_ok=True)
    count=0
    data, r_min, r_max = data_loading(data_path, args.target)
    total_rows = len(data)
    batch_rows = args.group
    total_batches = total_rows // batch_rows
    if total_rows % batch_rows != 0:
        total_batches += 1
    print(f"总批次:{total_batches}")
    for i in range(0, total_rows, batch_rows):
        batch_data = data[i:i + batch_rows]
        batch_file_name = f"batch_{count + 1}.csv"
        batch_file_path = os.path.join(data_next_path, batch_file_name)
        # 保存每个批次的数据到文件
        batch_data.to_csv(batch_file_path, index=False)
        print(f"保存批次 {count + 1} 数据到: {batch_file_path}")
        count += 1


