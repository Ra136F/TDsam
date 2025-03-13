import pandas as pd
import numpy as np
import time
import glob
import os





def TDSam(D, r, n):
    """
    核心算法：对时间序列数据进行采样，提取关键点。
    Parameters:
    D (np.ndarray): 输入数据数组
    r (float): 阈值，用于确定关键点
    D_kp (np.ndarray): 输出关键点数组
    n (int): 数据长度
    """
    result=list()
    for i in range(n):
        if i == 0:
            result.append(D[i])
        elif i == n - 1:
            result.append(D[i])
        elif 1 <= i < n - 1:
            if D[i - 1] < D[i] > D[i + 1] or D[i - 1] > D[i] < D[i + 1]:
                if D[i] > D[i + 1] + r and D[i] > D[i - 1] + r:
                    result.append(D[i])
                elif D[i] < D[i + 1] - r and D[i] < D[i - 1] - r:
                    result.append(D[i])
            elif abs(D[i + 1] - 2 * D[i] + D[i - 1]) > r:
                result.append(D[i])
    result=np.array(result)
    return result






