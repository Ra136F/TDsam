import  numpy as np

# 定义参数
total_data_points = 16  # 总数据点数
window_size = 6  # 窗口大小
predict_length = 2  # 预测长度
# 随机生成一个数据集
data_set = np.random.rand(total_data_points)
# 分割数据集的函数
def create_inout_sequences(input_data, tw, pre_len):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# 使用函数分割数据
X = create_inout_sequences(data_set, window_size, predict_length)
print("Input sequences (X):")
print(X)
