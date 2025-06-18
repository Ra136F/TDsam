import pandas as pd
import numpy as np
import time

def TDSam(data,  r):
    # 从 DataFrame 提取指定列的数据作为列表
    D = data['OT'].tolist()
    D_kp = [(D[0], D[-1])]  # 初始输出列表，包含第一个和最后一个元素

    for i in range(1, len(D) - 1):
        if D[i - 1] < D[i] > D[i + 1] or D[i - 1] > D[i] < D[i + 1]:  # 检查是否为非单调序列
            if D[i] > D[i + 1] + r and D[i] > D[i - 1] + r:
                D_kp.append(D[i])
            elif D[i] < D[i + 1] - r and D[i] < D[i - 1] - r:
                D_kp.append(D[i])
            elif abs(D[i + 1] - 2 * D[i] + D[i - 1]) > r:
                D_kp.append(D[i])

    return D_kp


data=pd.read_csv('./test/ETTh1.csv',index_col=['date'], parse_dates=['date'],usecols=['date','OT'])
diff=data['OT'].max()-data['OT'].min()
adjacent_diffs =data['OT'].diff().abs().iloc[1:]
min_adjacent_diff = adjacent_diffs[adjacent_diffs != 0].min()
print(diff)
print(min_adjacent_diff)
start_time = time.time()
D_kp=TDSam(data,0.28)
end_time = time.time()
print(len(D_kp))
print(len(data))
execution_time = end_time - start_time
print(f"TDSam 算法执行时间: {execution_time:.6f} 秒")