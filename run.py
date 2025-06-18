import argparse
from datetime import datetime
import time

from all2cloud import all_send
from simplets2cloud import sim_send
from xender2cloud import xender_send

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-method', type=str, default='s', help="传输方式")
    parser.add_argument('-data_name', type=str, default='oil-well', help="数据集名称")
    parser.add_argument('-target', type=str, default='T-JUS-CKP', help="目标特征")
    parser.add_argument('-lambda_value', type=float, default=0.001, help="采样率")
    parser.add_argument('-mode', type=int, default=0, help="[0,1],不适用GPU、使用GPU")
    parser.add_argument('-ip', type=str, default='10.12.54.122', help="IP地址")
    parser.add_argument('-port', type=str, default='5002', help="端口")
    parser.add_argument('-ratio', type=float, default=0.05, help="比例")
    args = parser.parse_args()

    start_time = time.time()
    if args.method == 'simplets' or args.method == 's':
        sim_send(args)
    elif args.method == 'xender' or args.method == 'x':
        xender_send(args)
    elif args.method == 'all' or args.method == 'a':
        all_send(args)
    end_time = time.time()
    duration = end_time - start_time
    print(f"传输完成,共花费:{duration: .4f} s")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 格式化日志条目
    log_entry = f"[{timestamp}] 传输方法: {args.method}, 耗时: {duration:.4f} 秒\n"
    save_path=f"./result/{args.data_name}_{args.method}.txt"
    with open(save_path, "w") as f:
        f.write(log_entry)

