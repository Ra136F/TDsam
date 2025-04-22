import numpy as np
from typing import List


class TDSampler:  # 客户端采样算法代码
    def __init__(self, initial_lambda: float = 1.0):
        self.lambda_val = initial_lambda
        self.delta = 0.0

    def find_key_points(self, data: np.ndarray) -> List[int]:
        """sample"""
        key_indices = [0, len(data) - 1]

        for i in range(1, len(data) - 1):
            self._update_delta(data, i)

            if self._is_peak(data, i) or self._is_valley(data, i) or self._has_high_curvature(data, i):
                key_indices.append(i)

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