import numpy as np
from typing import List

from numba import njit, prange


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
            is_key_point = self._compute_key_points(data, self.lambda_val)
            key_indices.extend(i for i in range(1, n - 1) if is_key_point[i])


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

    @staticmethod
    @njit(parallel=True)
    def _compute_key_points(data: np.ndarray, lambda_val: float) -> np.ndarray:
        """并行计算所有点是否为关键点（Numba加速）"""
        n = len(data)
        # 创建布尔数组标记关键点
        key_flags = np.zeros(n, dtype=np.bool_)

        # 并行遍历所有内部点（首尾点已单独处理）
        for i in prange(1, n - 1):
            # 检查是否为峰值
            is_peak = (data[i] > data[i - 1] + lambda_val) and (data[i] > data[i + 1] + lambda_val)

            # 检查是否为谷值
            is_valley = (data[i] < data[i - 1] - lambda_val) and (data[i] < data[i + 1] - lambda_val)

            # 检查是否为高曲率点
            curvature = np.abs(data[i + 1] - 2 * data[i] + data[i - 1])
            is_high_curvature = curvature > lambda_val

            # 任何条件满足则标记为关键点
            key_flags[i] = is_peak or is_valley or is_high_curvature

        return key_flags