import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dtw_distance(seq1, seq2):
    """
    计算两个序列之间的 DTW 距离
    :param seq1: 序列 1，数组形式
    :param seq2: 序列 2，数组形式
    :return: DTW 距离
    """
    len1, len2 = len(seq1), len(seq2)
    dtw_matrix = np.zeros((len1 + 1, len2 + 1))

    # 初始化 DTW 矩阵
    dtw_matrix[0, :] = float('inf')
    dtw_matrix[:, 0] = float('inf')
    dtw_matrix[0, 0] = 0

    # 动态规划填充 DTW 矩阵
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            # 计算最小的累积距离
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # 插入
                dtw_matrix[i, j - 1],    # 删除
                dtw_matrix[i - 1, j - 1] # 匹配
            )

    # 返回最终的 DTW 距离
    return dtw_matrix[len1, len2]





