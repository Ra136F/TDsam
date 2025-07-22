import numpy as np


class AdaptiveCUSUM:
    def __init__(self, k=6, drift_k=0.5, min_sigma=0.1, alpha=0.1, min_segment_length=20):
        self.k = k  # 阈值乘数
        self.drift_k = drift_k  # 漂移参数乘数
        self.min_sigma = min_sigma  # 最小标准差下限
        self.alpha = alpha  # 标准差指数平滑参数
        self.min_segment_length = min_segment_length  # 最小分段长度

        # 重置内部状态
        self.reset()

    def reset(self):
        """重置检测器状态"""
        self.cusum_upper = 0  # 上累积和
        self.cusum_lower = 0  # 下累积和
        self.current_mean = None  # 当前段均值
        self.current_sigma = self.min_sigma  # 当前段标准差
        self.segment_length = 0  # 当前段数据点数量
        self.change_points = []  # 检测到的变化点列表
        self.data_buffer = []  # 当前段数据缓冲区
        self.threshold_history = []  # 阈值变化历史（用于分析）
        self.sigma_history = []  # 标准差变化历史（用于分析）

    def update(self, new_value):
        """
        处理新的数据点
        返回:
        - change_detected: 是否检测到变化点
        - change_position: 变化点位置（数据索引）
        """
        change_detected = False
        change_position = None

        # 将新值添加到缓冲区
        self.data_buffer.append(new_value)
        self.segment_length += 1

        # 初始化（第一个数据点）
        if self.current_mean is None:
            self.current_mean = new_value
            self.threshold_history.append(self.k * self.min_sigma)
            self.sigma_history.append(self.min_sigma)
            return False, None

        # 计算偏差
        deviation = new_value - self.current_mean

        # 更新标准差（使用指数加权移动平均）
        prev_sigma = self.current_sigma
        if self.segment_length == 2:
            # 对于第二个点，标准差为绝对偏差
            abs_dev = abs(deviation)
            self.current_sigma = max(abs_dev, self.min_sigma)
        else:
            # 指数加权标准差
            abs_dev = abs(deviation)
            weighted_dev = self.alpha * abs_dev + (1 - self.alpha) * prev_sigma
            self.current_sigma = max(weighted_dev, self.min_sigma)

        # 计算动态阈值和漂移参数
        dynamic_threshold = self.k * self.current_sigma
        dynamic_drift = self.drift_k * self.current_sigma

        # 更新累积和
        self.cusum_upper = max(0, self.cusum_upper + deviation - dynamic_drift)
        self.cusum_lower = max(0, self.cusum_lower - deviation - dynamic_drift)

        # 保存历史数据用于分析
        self.threshold_history.append(dynamic_threshold)
        self.sigma_history.append(self.current_sigma)

        # 检查是否检测到变化点
        if self.cusum_upper > dynamic_threshold or self.cusum_lower > dynamic_threshold:
            # 确保分段长度满足要求
            if self.segment_length >= self.min_segment_length:
                change_detected = True
                change_position = len(self.data_buffer) - 1
                self.change_points.append(change_position)

                # 重置状态
                self._start_new_segment()

        # 更新当前段均值（在线更新）
        self._update_mean(new_value)

        return change_detected, change_position

    def _update_mean(self, new_value):
        """在线更新均值（指数加权平均）"""
        if self.segment_length == 2:
            # 第二个点直接平均
            self.current_mean = (self.data_buffer[0] + new_value) / 2
        else:
            # 指数加权平均
            self.current_mean = self.alpha * new_value + (1 - self.alpha) * self.current_mean

    def _start_new_segment(self):
        """开始新分段"""
        # 使用最后几个点初始化新段
        init_points = self.data_buffer[-min(5, len(self.data_buffer)):]

        # 重置统计量
        self.cusum_upper = 0
        self.cusum_lower = 0
        self.current_mean = np.mean(init_points) if len(init_points) > 0 else 0
        self.segment_length = len(init_points)
        self.data_buffer = init_points.copy()

        # 重置标准差为初始最小标准差
        self.current_sigma = self.min_sigma

    def get_current_threshold(self):
        """获取当前阈值"""
        if len(self.threshold_history) == 0:
            return self.k * self.min_sigma
        return self.threshold_history[-1]
