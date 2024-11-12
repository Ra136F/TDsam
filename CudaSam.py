from numba import cuda
import pandas as pd
import time
import numpy as np

@cuda.jit
def TDSam_cuda(D, r, D_kp, n):
    i = cuda.grid(1)

    if 1 <= i < n - 1:
        # 判断是否为极大值或极小值
        if (D[i - 1] < D[i] > D[i + 1] or D[i - 1] > D[i] < D[i + 1]):
            if D[i] > D[i + 1] + r and D[i] > D[i - 1] + r:
                D_kp[i] = D[i]
            elif D[i] < D[i + 1] - r and D[i] < D[i - 1] - r:
                D_kp[i] = D[i]
            elif abs(D[i + 1] - 2 * D[i] + D[i - 1]) > r:
                D_kp[i] = D[i]
        else:
            D_kp[i] = 0  # 不符合条件，填充为 0（可以视为无效点）

data=pd.read_csv('./test/ETTh1.csv',index_col=['date'], parse_dates=['date'],usecols=['date','OT'])
D = data['OT'].values
n = len(D)

# 创建空的 NumPy 数组来存储结果
D_kp = np.zeros_like(D)

threads_per_block = 32
blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

start_time = time.time()
TDSam_cuda[blocks_per_grid, threads_per_block](D, 0.28, D_kp, n)
cuda.synchronize()
end_time = time.time()

execution_time = end_time - start_time
print(f"TDSam CUDA 加速算法执行时间: {execution_time:.6f} 秒")

# 将非零结果转换为 DataFrame 并显示
result_df = pd.DataFrame({'column_name': D_kp[D_kp != 0]})
print(result_df)