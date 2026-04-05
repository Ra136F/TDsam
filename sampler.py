import numpy as np
from typing import List

from numba import njit, prange
from numba import cuda, njit, prange


class TDSampler:  # 客户端采样算法代码
    def __init__(self, initial_lambda: float = 1.0,gpu=0):
        self.lambda_val = initial_lambda
        self.delta = 0.0
        self.gpu = gpu

    def find_key_points(self, data: np.ndarray) -> List[int]:
        """sample"""
        key_indices = [0, len(data) - 1]
        n = len(data)
        if self.gpu==0:
            for i in range(1, len(data) - 1):
                self._update_delta(data, i)

                if self._is_peak(data, i) or self._is_valley(data, i) or self._has_high_curvature(data, i):
                    key_indices.append(i)
        else:
            # is_key_point = self._compute_key_points(data, self.lambda_val)
            # key_indices.extend(i for i in range(1, n - 1) if is_key_point[i])
            d_data = cuda.to_device(data)

            # 创建一个空的布尔数组，用来保存每个点是否是关键点
            d_key_flags = cuda.device_array(n, dtype=np.bool_)

            # 计算关键点
            threads_per_block = 256
            blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

            # 启动 GPU 内核
            self._compute_key_points[blocks_per_grid, threads_per_block](d_data, self.lambda_val, d_key_flags)

            # 将结果从 GPU 内存拷贝回主机
            key_flags = d_key_flags.copy_to_host()

            # 根据计算结果提取关键点的索引
            key_indices.extend(i for i in range(1, n - 1) if key_flags[i])


        return sorted(key_indices)

    def _update_delta(self, data: np.ndarray, i: int):
        diff = abs(data[i] - data[i - 1])
        if diff > 0 and (self.delta == 0 or diff < self.delta):
            self.delta = diff

    def _is_peak(self, data: np.ndarray, i: int) -> bool:
        return bool(data[i] > (data[i - 1] + self.lambda_val) and data[i] > (data[i + 1] + self.lambda_val))

    def _is_valley(self, data: np.ndarray, i: int) -> bool:
        return bool(data[i] < (data[i - 1] - self.lambda_val) and
                    data[i] < (data[i + 1] - self.lambda_val))

    def _has_high_curvature(self, data: np.ndarray, i: int) -> bool:
        return abs(data[i + 1] - 2 * data[i] + data[i - 1]) > self.lambda_val

    # @staticmethod
    # @njit(parallel=True)
    # def _compute_key_points(data: np.ndarray, lambda_val: float) -> np.ndarray:
    #     """并行计算所有点是否为关键点（Numba加速）"""
    #     n = len(data)
    #     # 创建布尔数组标记关键点
    #     key_flags = np.zeros(n, dtype=np.bool_)
    #
    #     # 并行遍历所有内部点（首尾点已单独处理）
    #     for i in prange(1, n - 1):
    #         # 检查是否为峰值
    #         is_peak = (data[i] > data[i - 1] + lambda_val) and (data[i] > data[i + 1] + lambda_val)
    #
    #         # 检查是否为谷值
    #         is_valley = (data[i] < data[i - 1] - lambda_val) and (data[i] < data[i + 1] - lambda_val)
    #
    #         # 检查是否为高曲率点
    #         curvature = np.abs(data[i + 1] - 2 * data[i] + data[i - 1])
    #         is_high_curvature = curvature > lambda_val
    #
    #         # 任何条件满足则标记为关键点
    #         key_flags[i] = is_peak or is_valley or is_high_curvature
    #
    #     return key_flags
    @staticmethod
    @cuda.jit
    def _compute_key_points(data: np.ndarray, lambda_val: float, key_flags: np.ndarray):
        """
        GPU 内核：并行计算每个数据点是否为关键点
        1. 峰值: 大于前后点加上 lambda_val
        2. 谷值: 小于前后点减去 lambda_val
        3. 高曲率点：绝对二阶差值大于 lambda_val
        """
        idx = cuda.grid(1)  # 获取当前线程的唯一索引
        n = len(data)

        if 1 <= idx < n - 1:  # 确保索引在有效范围内
            # 判断是否为峰值
            is_peak = (data[idx] > data[idx - 1] + lambda_val) and (data[idx] > data[idx + 1] + lambda_val)

            # 判断是否为谷值
            is_valley = (data[idx] < data[idx - 1] - lambda_val) and (data[idx] < data[idx + 1] - lambda_val)

            # 判断是否为高曲率点
            curvature = abs(data[idx + 1] - 2 * data[idx] + data[idx - 1])
            is_high_curvature = curvature > lambda_val

            # 任何条件满足则标记为关键点
            key_flags[idx] = is_peak or is_valley or is_high_curvature



class RandomSampler:
    def __init__(self, sample_ratio: float = 0.1, gpu=0):
        """
        随机采样器构造函数
        :param sample_ratio: 采样比例 (0.0 ~ 1.0)
        :param gpu: 预留参数，保持接口一致性
        """
        assert 0 <= sample_ratio <= 1, "sample_ratio must be in [0.0, 1.0]"
        self.sample_ratio = sample_ratio
        self.gpu = gpu  # 保持接口兼容
        self.lambda_val = 0

    def find_key_points(self, data: np.ndarray) -> List[int]:
        """
        执行随机采样，返回关键点索引
        :param data: 输入数据数组
        :return: 排序后的关键点索引列表
        """
        n = len(data)
        # 处理边界情况
        if n == 0:
            return []
        if n == 1:
            return [0]
        # 始终包含首尾点
        key_indices = [0, n - 1]
        # 当数据点不足3个时直接返回
        if n <= 2:
            return key_indices
        # 计算需要采样的中间点数量（不包括首尾点）
        num_mid_points = n - 2
        sample_size = max(0, min(int(round(self.sample_ratio * num_mid_points)), num_mid_points))
        # 如果采样点数为0，只返回首尾点
        if sample_size == 0:
            return key_indices
        # 随机选择中间点索引
        mid_indices = list(range(1, n - 1))
        sampled_mid_indices = np.random.choice(mid_indices, size=sample_size, replace=False).tolist()

        key_indices.extend(sampled_mid_indices)
        return sorted(key_indices)



class RandomSampler2:
    def __init__(self, sample_prob: float = 0.3, gpu=0):
        """
        随机采样器构造函数
        :param sample_prob: 每个点被采样的概率 (0.0 ~ 1.0)
        :param gpu: 预留参数，保持接口一致性
        """
        assert 0 <= sample_prob <= 1, "sample_prob must be in [0.0, 1.0]"
        self.sample_prob = sample_prob
        self.gpu = gpu  # 保持接口兼容
        self.lambda_val = 0

    def find_key_points(self, data: np.ndarray) -> List[int]:
        """
        执行随机采样，返回关键点索引
        :param data: 输入数据数组
        :return: 排序后的关键点索引列表
        """
        n = len(data)
        # 处理边界情况
        if n == 0:
            return []
        if n == 1:
            return [0]
        # 始终包含首尾点
        key_indices = [0, n - 1]
        # 当数据点不足3个时直接返回
        if n <= 2:
            return key_indices
        # 遍历中间点，以概率sample_prob决定是否采样
        for i in range(1, n - 1):
            if np.random.rand() < self.sample_prob:
                key_indices.append(i)
        return sorted(key_indices)