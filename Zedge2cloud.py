#!/usr/bin/python3.7
import argparse
# Run command example
# $time python3.7 edge2cloud.py GCE put bigfiles adaptive 0 0 1 4 10 10

import os, sys
import platform
import hashlib
import json

import time
import datetime
import string

import multiprocessing
import threading, queue

import psutil


import operator, random


import requests

from datetime import datetime

from util import data_loading

# (upload, download)
SYSTEM_BANDWIDTH = (float(0.5), float(0.5))

metadata_db = 'metadata.json'
suppoted_compressors = ('gzip', 'bzip2', 'lzma', 'zstd')
#compress_factors = {'gzip' : '-9', 'bzip2' : '-1', 'lzma' : '-1', 'zstd' : '-13'}
compress_factors = {'gzip' : '-9', 'bzip2' : '-9', 'lzma' : '-9', 'zstd' : '-19'}
#compress_factors = {'gzip' : ' ', 'bzip2' : ' ', 'lzma' : '-1', 'zstd' : ' '}

transferthread_list = []
compressthread_list = []

compress_queue = queue.Queue()
compress_threads = int(1)
compressthread_exit = 0

transfer_queue = queue.Queue()
transfer_threads = int(1)

cores_num = int(1)

files_to_send = []
valid_file_count = 0
transfer_count = 0
timeout_sec = 10
retry = 0

lock = threading.Lock()

# For adaptive compressor selection
sample_db = 'sample.json'
sample_files = [ 'samples/0601.tar.sample',  'samples/0701.tar.sample',  'samples/Nixon.tif', 'samples/1201.tar.sample']
filegroups = [  ['bigfiles/0601.tar', 'bigfiles/0602.tar', 'bigfiles/0603.tar', 'bigfiles/0604.tar', 'bigfiles/0605.tar'], \
                ['bigfiles/0701.tar', 'bigfiles/0702.tar', 'bigfiles/0703.tar', 'bigfiles/0704.tar', \
                'bigfiles/0701_2.tar', 'bigfiles/0702_2.tar', 'bigfiles/0703_2.tar', 'bigfiles/0704_2.tar', \
                'bigfiles/0701_3.tar', 'bigfiles/0702_3.tar', 'bigfiles/0703_3.tar', 'bigfiles/0704_3.tar'], \
                [
                'bigfiles/1101_apollo16_earth_northamerica.tiff',\
                'bigfiles/1102_apollo17_earth.tiff',\
                'bigfiles/1103_clem_lake_victoria.tiff',\
                'bigfiles/1104_gal_moon_nims_41476.tiff',\
                'bigfiles/1105_hst_jupiter_aurorae.tiff',\
                'bigfiles/1106_jupiter_family.tiff'],
                [
                'bigfiles/1201.tar', 'bigfiles/1202.tar', 'bigfiles/1203.tar', 'bigfiles/1204.tar',\
                'bigfiles/1205.tar', 'bigfiles/1206.tar', 'bigfiles/1207.tar', 'bigfiles/1208.tar', 'bigfiles/1209.tar']
             ]


def group_transfer():
    global valid_file_count
    global transfer_count
    global timeout_sec
    global retry
    global compressthread_exit

    while True:
        if transfer_queue.empty():
            if compressthread_exit == compress_threads:
                break

        item = transfer_queue.get()

        print('Sending {}'.format(item))
        #SCP_Func(item)
        HTTP_Func(item)
        transfer_queue.task_done()
        transfer_count += 1
def group_transfer2(c_file):
    global valid_file_count
    global transfer_count




    item = c_file

    print('Sending {}'.format(item))
    # SCP_Func(item)
    HTTP_Func(item)
    transfer_queue.task_done()
    transfer_count += 1




def group_compress(compressOption):
    # net_if_addrs = psutil.net_if_addrs()
    #
    # # 打印所有的网络接口及其信息
    # for interface, addresses in net_if_addrs.items():
    #     print(f"Interface: {interface}")
    #     for address in addresses:
    #         print(f"  Address: {address.address}, Netmask: {address.netmask}, Broadcast: {address.broadcast}")
    global compressthread_exit
    while True:
        #try:
        #    item = compress_queue.get()
        if compress_queue.empty():
            lock.acquire()
            compressthread_exit += 1
            lock.release()
            break
        item = compress_queue.get()

        selectedCompressOption = None

        if compressOption == 'adaptive':
            # sampleid = getsample(item)
            sampleid = item
            print("-------------------------")
            selectedCompressOption = policy_engine(sampleid, suppoted_compressors, args.network, 1)
            print("-------------------------")
            print('Best compressor for {} based on sample {}: {}'.format(item, sampleid, selectedCompressOption))
        elif compressOption == 'random':
            selectedCompressOption = random.choice(suppoted_compressors)
            print('Random compressor for {}: {}'.format(item, selectedCompressOption))

        if selectedCompressOption:
            c_file = E2C_Compress(item, selectedCompressOption)
        else:
            c_file = E2C_Compress(item, compressOption)

        transfer_queue.put(c_file)
        compress_queue.task_done()

def group_compress2(file_path,compressOption):
    item = file_path
    selectedCompressOption = None
    if compressOption == 'adaptive':
        # sampleid = getsample(item)
        sampleid = item
        print("-------------------------")
        selectedCompressOption = policy_engine(sampleid, suppoted_compressors, args.network, 1)
        print("-------------------------")
        print('Best compressor for {} based on sample {}: {}'.format(item, sampleid, selectedCompressOption))
    elif compressOption == 'random':
        selectedCompressOption = random.choice(suppoted_compressors)
        print('Random compressor for {}: {}'.format(item, selectedCompressOption))

    if selectedCompressOption:
        c_file = E2C_Compress(item, selectedCompressOption)
    else:
        c_file = E2C_Compress(item, compressOption)
    return c_file

# folder must be a folder name
def group_put(folder, compressOption):
    global valid_file_count
    global transfer_count

    files_to_send = os.listdir(folder)
    files_to_send = sorted(files_to_send)
    for f in files_to_send:
        filepath = folder+'/'+f
        if os.path.isfile(filepath):
            compress_queue.put(filepath)
            valid_file_count += 1
    # compress_queue.put(folder)
    # valid_file_count += 1

    for i in range(compress_threads):
        ct = threading.Thread( target = group_compress, args = (compressOption,) )
        compressthread_list.append(ct)
        ct.start()

    for j in range(transfer_threads):
        tt = threading.Thread( target = group_transfer, args = ())
        transferthread_list.append(tt)
        tt.start()


if os.path.exists(metadata_db):
    try:
        metadata_dict = json.load(open(metadata_db,"r"))
    except:
        print('{} exists, but cannot be loaded'.format(metadata_db))
        sys.exit()
else:
    metadata_dict = {};

if os.path.exists(sample_db):
    try:
        sample_dict = json.load(open(sample_db,"r"))
    except:
        print('{} exists, but cannot be loaded'.format(sample_db))
        sys.exit()
else:
    sample_dict = {};


# if settings.google_drv == 1:
#     gauth = GoogleAuth()
#     # Try to load saved client credentials
#     gauth.LoadCredentialsFile("mycreds.txt")
#     if gauth.credentials is None:
#         # Authenticate if they're not there
#         gauth.LocalWebserverAuth()
#     elif gauth.access_token_expired:
#         # Refresh them if expired
#         gauth.Refresh()
#     else:
#         # Initialize the saved creds
#         gauth.Authorize()
#     # Save the current credentials to a file
#     gauth.SaveCredentialsFile("mycreds.txt")
#     drive = GoogleDrive(gauth)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# def E2C_List(fileIdPattern):
#     """Fucntion lists cloud files, the names of which match the given pattern
#
#     Args:
#         fileIdPattern: the filename pattern in string
#
#     Returns:
#         A list of filenames that matches the pattern.
#
#     """
#     file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
#     ret = []
#     for fid in file_list:
#         if fileIdPattern in fid['title']:
#             ret.append(fid['title'])
#     return ret

def Metadata_Dict_Get(fileId):
    return metadata_dict[fileId]

def Metadata_Dict_Insert(fileId, attributes):
    metadata_dict[fileId] = attributes

def Metadata_Dict_Delete(fileId):
    del metadata_dict[fileId]

def sample_Dict_Get(fileId):
    return sample_dict[fileId]

def sample_Dict_Insert(fileId, attributes):
    sample_dict[fileId] = attributes

def sample_Dict_Delete(fileId):
    del sample_dict[fileId]



def E2C_Compress(fileId, compressOption):
    o_size = os.path.getsize(fileId)
    c_fileId = fileId
    c_size = os.path.getsize(c_fileId)
    start_time = time.time()
    if compressOption in suppoted_compressors:
        if compressOption == 'gzip' :
            os.system('{} {} -f -k {}'.format(compressOption, compress_factors[compressOption], fileId))
            c_fileId =  fileId+'.gz'
        if compressOption == 'bzip2' :
            os.system('{} {} -f -k {}'.format(compressOption, compress_factors[compressOption], fileId))
            c_fileId = fileId+'.bz2'
        if compressOption == 'lzma' :
            os.system('{} {} -f -k {}'.format(compressOption, compress_factors[compressOption], fileId))
            c_fileId = fileId+'.lzma'
        if compressOption == 'zstd' :
            os.system('{} {} -f -k {}'.format(compressOption, compress_factors[compressOption], fileId))
            c_fileId = fileId+'.zst'

    c_size = os.path.getsize(c_fileId)
    if c_size:
        print('##### {}, compression ratio, {}'.format(compressOption, o_size/c_size))
    else:
        print('Compressed file {} has size 0'.format(c_fileId))

    print('##### Compression time, {}, seconds'.format(time.time()-start_time))
    return c_fileId


def E2C_Uncompress(fileId, compressOption):
    if compressOption in suppoted_compressors:
        os.system('{} -d -f {}'.format(compressOption, fileId))

# def DRV_Put (fileId, compressOption, secureOption, secureKey):
#     """Uploads a edge local file object to cloud storage.
#
#     Args:
#         fileId: the filename in string
#         compressOption: compress scheme. Options include gzip, bz2, lzma, adaptive, or none
#         secureOption: encryption schemes. Options include AES, RSA, TEE (to be implemented)
#         secureKey: the key used to encrypt/decrypt the file object
#
#     Returns:
#         Error code or None for success.
#
#     """
#
#     start_time = time.time()
#     # compress the file
#     localfile = E2C_Compress(fileId, compressOption)
#     print("{}, compression cost,{}, seconds ---".format(compressOption, time.time() - start_time))
#
#     start_time = time.time()
#     gfile = drive.CreateFile({'title' : localfile})
#     gfile.SetContentFile(localfile)
#     gfile.Upload() # Upload the file.
#     print("Edge to Cloud transfer cost,{}, seconds ---".format(time.time() - start_time))
#     print('title: %s, mimeType: %s' % (gfile['title'], gfile['mimeType']))
#     attributes = {'clouduri' : gfile['id'], 'compress' : compressOption, 'secure' : secureOption, 'key' : secureKey}
#     Metadata_Dict_Insert(fileId, attributes)
#
# def DRV_Delete (fileId, secureKey):
#     try:
#         gfile = drive.CreateFile({'id': Metadata_Dict_Get(fileId)})
#     except:
#         print('{} does not exist'.format(fileId))
#     gfile.Delete()
#     Metadata_Dict_Delete(fileId)



def cpu_idle_stat():
    core_cnt = psutil.cpu_count()
    percore_stat = psutil.cpu_times_percent(interval=0.01, percpu=True)
    total_idle = 0
    print('### Idle utilization in CPU power. 1 means a whole core is idle ###')
    for i in range(core_cnt):
        print('    >> Core {}, idle utilization, {}'.format(i, percore_stat[i].idle))
        total_idle += percore_stat[i].idle
        print('>>>> All core, aggregated idle utilization, {}'.format(total_idle))

    return total_idle/100


def idle_disk_bandwidth():
    # skip real tests
    return 100

    os.system('dd if=/dev/zero of=.test1 bs=512k count=20 oflag=direct')
    disk_read_rate = os.popen("dd if=.test1 of=.test2 bs=512k count=20 oflag=direct 2>&1 | awk '/copied/ {print $10}' ").read().rstrip("\n\r")
    print( 'disk_read_rate, {}, MB/s'.format(disk_read_rate) )
    return float(disk_read_rate)


def idle_net_bandwidth(nic):
    # Available network bandwidth test, no matter there is interference or not
    # pressure test results reflect real avaiable bandwidth
    # This does not reflect total network bandwith, because of interference

    global SYSTEM_BANDWIDTH

    # test interval
    period = 0.02
    counts = psutil.net_io_counters(pernic=True)
    old_bytes_sent = counts[nic].bytes_sent
    old_bytes_recv = counts[nic].bytes_recv
    time.sleep(period)
    counts = psutil.net_io_counters(pernic=True)
    new_bytes_sent = counts[nic].bytes_sent
    new_bytes_recv = counts[nic].bytes_recv
    used_upload_rate = (new_bytes_sent - old_bytes_sent) / period / 1024. / 1024.
    used_download_rate = (new_bytes_recv - old_bytes_recv) / period / 1024. / 1024.
    # for 4 transfer threads, one thread can have 1/4 bandwidth
    avail_upload_rate = max(0.1, args.upload_bandwidth - used_upload_rate)
    avail_download_rate = max(0.1,args.download_bandwidth - used_download_rate)
    print('upload_rate, {}, MB/s, download_rate, {}, MB/s'.format(avail_upload_rate, avail_download_rate))

    return (avail_upload_rate, avail_download_rate)


def compress_rate(fileId, compressOption):
    if compressOption not in suppoted_compressors:
        return None, None
    o_size = os.path.getsize(fileId)
    c_fileId = fileId
    if compressOption == 'gzip' :
        start_time = time.time()
        print("执行gzip压缩")
        os.system('{} -9 -f -k {}'.format(compressOption, fileId))
        end_time = time.time()
    elif compressOption == 'bzip2' :
        print("执行bzip2压缩")
        start_time = time.time()
        os.system('{} -1 -f -k {}'.format(compressOption, fileId))
        end_time = time.time()
    elif compressOption == 'lzma' :
        print("执行lzma压缩")
        start_time = time.time()
        os.system('{} -1 -f -k {}'.format(compressOption, fileId))
        end_time = time.time()
    elif compressOption == 'zstd' :
        print("执行zstd压缩")
        start_time = time.time()
        os.system('{} -13 -f -k {}'.format(compressOption, fileId))
        end_time = time.time()
    else:
        start_time = time.time()
        os.system('{} -f -k {}'.format(compressOption, fileId))
        end_time = time.time()

    if compressOption == 'gzip':
        c_fileId =  fileId+'.gz'
    if compressOption == 'bzip2':
        c_fileId = fileId+'.bz2'
    if compressOption == 'lzma':
        c_fileId = fileId+'.lzma'
    if compressOption == 'zstd':
        c_fileId = fileId+'.zst'

    c_size = os.path.getsize(c_fileId)
    c_ratio = o_size/c_size
    c_rate = o_size / 1024. / 1024. / (end_time - start_time)
    print('{}, compression ratio, {}, compression rate, {}, MB/s'.format(compressOption, c_ratio, c_rate))

    return c_rate, c_ratio


def getsample(fileId):
    print('tag:'+fileId)
    for i in range(len(filegroups)):
        print(filegroups[i])
        if fileId in filegroups[i]:
            return sample_files[i]
    return None

# Using one file to predict is not accurate
# file-level selection, only one core can be used for one file
def policy_engine(sampleId, compressorList, nic, core_num):
    # Total idle cpu utilization, disk bandwidth, network bandwidth
    print(f"sampleId: {sampleId}")
    start_time = time.time()
    #A_1_n = cpu_idle_stat()
    A_1_n = 0.99
    core_num = cores_num
    A_1_n = min(A_1_n, core_num)
    print('$$$$$ CPU profile time, {}, seconds'.format(time.time() - start_time))

    start_time = time.time()
    D = idle_disk_bandwidth()
    print('$$$$$ Disk profile time, {}, seconds'.format(time.time() - start_time))

    start_time = time.time()
    B = idle_net_bandwidth(nic)[0]
    print('$$$$$ network profile time, {}, seconds'.format(time.time() - start_time))

    print('Total idle cpu utilization, {} cores, disk bandwidth,{} MB/s, network bandwidth, {} MB/s'.format(A_1_n, D, B))

    res = {}
    start_time = time.time()
    for compressor in compressorList: 
        if compressor == 'gzip':
            c_sampleId =  sampleId+'.gz'
        if compressor == 'bzip2':
            c_sampleId = sampleId+'.bz2'
        if compressor == 'lzma':
            c_sampleId = sampleId+'.lzma'
        if compressor == 'zstd':
            c_sampleId = sampleId+'.zst'

        if sampleId in sample_dict:
            if compressor in sample_dict[sampleId]:
                T_c, R_c = sample_dict[sampleId][compressor]
            else:
                T_c, R_c = compress_rate(sampleId, compressor)
                sample_dict[sampleId][compressor] = (T_c, R_c)
        else:
            T_c, R_c = compress_rate(sampleId, compressor)
            sample_dict[sampleId] = {compressor: (T_c, R_c)}
        
        TaC = T_c * A_1_n
        TLC = min(TaC, D)
        TnC = min(TLC, B * R_c)
        print('compressor {}, T_c {}, R_c {}, B {}, TaC {}, TLC {}, TnC {}'.format(compressor, T_c, R_c, B, TaC, TLC, TnC))
        res[compressor] = [TnC]

        #os.system('rm {}.*'.format(sampleId))
    # return compressor name
    #print('$$$$$ Compression profile time, {}, seconds'.format(time.time() - start_time))

    return max(res.items(), key=operator.itemgetter(1))[0]


# Using one file to predict is not accurate
# file-level selection, only one core can be used for one file
def max_compressratio_engine(sampleId, compressorList, nic, core_num):
    # Total idle cpu utilization, disk bandwidth, network bandwidth

    res = {}
    for compressor in compressorList:
        if compressor == 'gzip':
            c_sampleId =  sampleId+'.gz'
        if compressor == 'bzip2':
            c_sampleId = sampleId+'.bz2'
        if compressor == 'lzma':
            c_sampleId = sampleId+'.lzma'
        if compressor == 'zstd':
            c_sampleId = sampleId+'.zst'

        if sampleId in sample_dict:
            if compressor in sample_dict[sampleId]:
                T_c, R_c = sample_dict[sampleId][compressor]
            else:
                T_c, R_c = compress_rate(sampleId, compressor)
                sample_dict[sampleId][compressor] = (T_c, R_c)
        else:
            T_c, R_c = compress_rate(sampleId, compressor)
            sample_dict[sampleId] = {compressor: (T_c, R_c)}

        res[compressor] = R_c

        os.system('rm {}.*'.format(sampleId))
    # return compressor name
    #print('$$$$$ Compression profile time, {}, seconds'.format(time.time() - start_time))

    return max(res.items(), key=operator.itemgetter(1))[0]


def HTTP_Func(localfile):
    global SERVER_URL
    # 打开文件并作为multipart/form-data请求发送
    with open(localfile, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://10.12.54.122:5001/upload', files=files)
        
        if response.status_code == 200:
            print("文件 {} 上传成功。".format(localfile))
            # 获取当前时间

            end_time = time.time()
            duration = end_time - start_time
            print(f"传输完成,共花费:{duration:.4f} s")

        else:
            print("文件 {} 上传失败。HTTP状态码: {}".format(localfile, response.status_code))




# Extreme compression ratio tests
def main2():
    platform = sys.argv[1]
    command = sys.argv[2]
    fileIds = os.listdir(sys.argv[3])

    start_time = time.time()
    if command == 'put':
        compressOption = sys.argv[4]
        secureOption = sys.argv[5]
        secureKey = sys.argv[6]
        if platform == 'DRV':
            # DRV_Put (fileId, compressOption, secureOption, secureKey)
            print('DRV')
        elif platform == 'GCE':
            print('PARAMETERS: ', sys.argv[3], compressOption)
            group_put(sys.argv[3], compressOption)
        elif platform == 'HTTP':  # HTTP平台情况
            print('参数: ', sys.argv[3], compressOption)
            group_put(sys.argv[3], compressOption)
        else:
            print('Platform ERROR: argv[1] must be either DRV or GCE or HTTP')

    for t in compressthread_list:
        if t.is_alive():
            t.join()

    print('Final Statistics, group_put {} cost, {}, seconds, valid_file_count: {}, success transfer_count: {},'.format(sys.argv[3], time.time() - start_time, valid_file_count, transfer_count))
    
    

# Shortest transfer time tests
def main():
    print("main")
    data_path="./data/"+args.data_name+"/"
    global valid_file_count
    global transfer_count

    files_to_send = os.listdir(data_path)
    files_to_send = sorted(files_to_send)
    for f in files_to_send:
        filepath = data_path+'/'+f
        if os.path.isfile(filepath):
            # compress_queue.put(filepath)
            c_file=group_compress2(filepath,args.compressOption)
            group_transfer2(c_file)


    # group_put(data_path, args.compressOption)




    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='客户端传输')
    parser.add_argument('-data_name', type=str, default='energy', help="数据集名称")
    parser.add_argument('-target', type=str, default='T1', help="目标特征")
    parser.add_argument('-ip', type=str, default='10.12.54.122', help="IP地址")
    parser.add_argument('-port', type=str, default='5001', help="端口")
    parser.add_argument('-ratio', type=float, default=0.002, help="比例")
    parser.add_argument('-group', type=int, default=300, help='分组')
    parser.add_argument('-compressOption', type=str, default='adaptive', help="压缩策略")
    parser.add_argument('-network', type=str, default='wlan0', help='网卡名称')
    parser.add_argument("-upload_bandwidth", type=float, default=0.1, help="上传带宽")
    parser.add_argument("-download_bandwidth", type=float, default=0.1, help="下载带宽")
    args = parser.parse_args()
    start_time = time.time()
    main()

