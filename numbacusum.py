from numba import njit, float64, int64, types
import numpy as np

class NumbaCUSUM:
    def __init__(self, k=6, drift_k=0.5, min_sigma=0.1, alpha=0.1, min_segment_length=20):
        self.k = k
        self.drift_k = drift_k
        self.min_sigma = min_sigma
        self.alpha = alpha
        self.min_segment_length = min_segment_length

        # 初始化状态
        self.reset()

        # 预编译 Numba 函数
        _ = _cusum_update_numba(
            0.0,
            (0.0, 0.0, 0.0, 0.0, 0),
            self.k, self.drift_k, self.min_sigma, self.alpha, self.min_segment_length
        )

    def reset(self):
        """重置检测器状态"""
        # Numba 兼容状态元组
        self.state = (0.0, 0.0, np.nan, self.min_sigma, 0)
        self.change_points = []
        self.last_global_index = 0  # 用于记录变化点索引

    def update(self, new_value):
        """处理新数据点"""
        # 调用预编译的 Numba 函数
        self.state, change_detected, _ = _cusum_update_numba(
            float(new_value),
            self.state,
            self.k,
            self.drift_k,
            self.min_sigma,
            self.alpha,
            self.min_segment_length
        )

        # 记录变化点
        if change_detected:
            # 变化位置为当前点（全局索引）
            change_position = self.last_global_index
            self.change_points.append(change_position)
            self.last_global_index += 1
            return True, change_position

        self.last_global_index += 1
        return False, None

    def batch_update(self, values):
        """批量处理数据点"""
        results = []
        for val in values:
            self.state, change_detected, _ = _cusum_update_numba(
                float(val),
                self.state,
                self.k,
                self.drift_k,
                self.min_sigma,
                self.alpha,
                self.min_segment_length
            )
            if change_detected:
                # 变化位置为当前处理点索引
                change_position = self.last_global_index
                self.change_points.append(change_position)
                results.append(True)
            else:
                results.append(False)
            self.last_global_index += 1
        return results




# 定义 Numba 类型签名
state_type = types.Tuple((
    float64,  # cusum_upper
    float64,  # cusum_lower
    float64,  # current_mean
    float64,  # current_sigma
    int64,  # segment_length
))


@njit([
    types.Tuple((state_type, types.boolean, float64))(
        float64,  # new_value
        state_type,  # current_state
        float64,  # k
        float64,  # drift_k
        float64,  # min_sigma
        float64,  # alpha
        int64  # min_segment_length
    )
], cache=True, fastmath=True)
def _cusum_update_numba(new_value, state, k, drift_k, min_sigma, alpha, min_segment_length):
    """
    Numba 加速的核心计算函数
    返回: (新状态, 变化检测标志, 动态阈值)
    """
    cusum_upper, cusum_lower, current_mean, current_sigma, segment_length = state

    # 初始化处理 (第一个数据点)
    if segment_length == 0:
        new_cusum_upper = 0.0
        new_cusum_lower = 0.0
        new_current_mean = new_value
        new_current_sigma = min_sigma
        new_segment_length = 1
        change_detected = False
        dynamic_threshold = k * min_sigma
        return ((new_cusum_upper, new_cusum_lower, new_current_mean,
                 new_current_sigma, new_segment_length),
                change_detected, dynamic_threshold)

    # 计算偏差
    deviation = new_value - current_mean

    # 更新标准差
    if segment_length == 1:  # 第二个点特殊处理
        abs_dev = abs(deviation)
        new_current_sigma = max(abs_dev, min_sigma)
    else:
        abs_dev = abs(deviation)
        # 指数加权移动标准差
        new_current_sigma = max(alpha * abs_dev + (1 - alpha) * current_sigma, min_sigma)

    # 计算动态参数
    dynamic_threshold = k * new_current_sigma
    dynamic_drift = drift_k * new_current_sigma

    # 更新累积和
    new_cusum_upper = max(0.0, cusum_upper + deviation - dynamic_drift)
    new_cusum_lower = max(0.0, cusum_lower - deviation - dynamic_drift)

    # 检查变化点
    change_detected = False
    if (new_cusum_upper > dynamic_threshold or
        new_cusum_lower > dynamic_threshold) and segment_length >= min_segment_length:
        change_detected = True
        # 重置状态（但保持当前值）
        new_cusum_upper = 0.0
        new_cusum_lower = 0.0
        new_segment_length = 1
        new_current_mean = new_value
        new_current_sigma = min_sigma
    else:
        # 更新均值（指数加权）
        new_current_mean = alpha * new_value + (1 - alpha) * current_mean
        new_segment_length = segment_length + 1

    new_state = (new_cusum_upper, new_cusum_lower, new_current_mean,
                 new_current_sigma, new_segment_length)

    return new_state, change_detected, dynamic_threshold