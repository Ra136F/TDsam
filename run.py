import argparse
import asyncio
import os
from datetime import datetime
import time

from SamplerTest import test, local_fenlei_cusum, local_fenlei_guding
from all2cloud import all_send
from simplets2cloud import sim_send
from util import init_args
from xender2cloud import xender_send, fenlei_send, fenlei_send2, fenlei_send3, fenlei_send4, \
    xender_pipeline_send2, xender_send_async

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-method', type=str, default='x', help="传输方式(全传输(all、a) xender(x) 固定窗口(guding、g) cusum(c))")
    parser.add_argument('-data_name', type=str, default='rain', help="数据集名称")
    parser.add_argument('-target', type=str, default='T1', help="目标特征")
    parser.add_argument('-lambda_value', type=float, default=0.25, help="采样率")
    parser.add_argument("-start_ori_time", type=int, default=0, help="开始传输原始数据的时间")
    parser.add_argument("-second_lambda",type=float,default=0.0,help="降低后的采样率")
    parser.add_argument('-aware',type=int,default=1,help="[0,1],内容不感知、内容感知")
    parser.add_argument('-mode', type=int, default=0, help="[0,1],不适用GPU、使用GPU")
    parser.add_argument('-ip', type=str, default='10.12.54.122', help="IP地址")
    parser.add_argument('-port', type=str, default='5002', help="端口")
    parser.add_argument('-ratio', type=float, default=0.002, help="比例")
    parser.add_argument('-group', type=int, default=200, help='分组')
    parser.add_argument('-sampler', type=str, default='xender', help="采样器(random or xender)")
    parser.add_argument("-k", type=int, default=10, help="k")
    parser.add_argument("-segment_length", type=int, default=200, help="cusum最小长度")
    args = parser.parse_args()

    start_time = time.time()

    args=init_args(args)
    if args.method == 'simplets' or args.method == 's':#无作用
        sim_send(args)
    elif args.method == 'xender' or args.method == 'x':#xender
        # xender_send(args)
        xender_send_async(args)
        # xender_pipeline_send2(args)
    elif args.method == 'all' or args.method == 'a':#全传输
        all_send(args)
    elif args.method == 'guding' or args.method == 'g':#guding
        fenlei_send(args)
    elif args.method == 'c' or args.method == 'cusum':#cusum
        fenlei_send4(args)
    else:#测试
        local_fenlei_guding(args)#
    end_time = time.time()
    duration = end_time - start_time
    print(f"传输完成,共花费:{duration: .4f} s")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 格式化日志条目
    log_entry = f"[{timestamp}] 传输方法: {args.method}, 耗时: {duration:.4f} 秒\n"
    file_path = f"./result/{args.data_name}/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    save_path=f"./result/{args.data_name}/gpu-{args.mode}-{args.method}-{args.group}-{timestamp}.txt"
    with open(save_path, "w") as f:
        f.write(log_entry)

