"""三通道、共享对齐路径的受限 DTW。

每个时刻是一行 [标准化速度, 标准化纵向加速度, 标准化横向加速度]。
先寻找累积代价最小的对齐路径，再用该路径长度归一化：
    distance = sqrt(最小累积代价 / 所选路径的匹配点对数)
注意：这不是直接寻找“平均代价最小”的路径，也没有单独加入通行时长差。
Numba 仅加速下面的 Python 数值循环，不使用神经网络或近似 DTW。
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True)
def cost_and_length(a, b, band):
    """动态规划内核；返回最小累积代价与对应路径长度。

    a 为 n×3，b 为 m×3，点数可以不同；调用前应检查数据合法性。
    只保留动态规划表的上一行和当前行，减少计算每一对车辆时的内存。
    """
    n, m = len(a), len(b)
    previous = np.full(m + 1, np.inf)
    current = np.full(m + 1, np.inf)
    previous_length = np.zeros(m + 1, np.int32)
    current_length = np.zeros(m + 1, np.int32)
    previous[0] = 0.0

    for i in range(n):
        current[:] = np.inf
        current_length[:] = 0

        # 比较相对经过时间进度，而不是要求两车具有相同的实际秒数。
        # 例如 band=0.1：A 在 40% 进度时，只能对应 B 的约 30%～50%。
        fraction = i / (n - 1)
        low = max(0, int(np.ceil((fraction - band) * (m - 1) - 1e-9)))
        high = min(m - 1, int(np.floor((fraction + band) * (m - 1) + 1e-9)))

        for j in range(low, high + 1):
            # 三种前驱分别表示：双方都前进一步、只推进 A、只推进 B。
            # 代价相同时优先匹配点对更少的路径，以保证选择规则可重复。
            best = previous[j]
            length = previous_length[j]
            if previous[j + 1] < best or (
                previous[j + 1] == best and previous_length[j + 1] < length
            ):
                best, length = previous[j + 1], previous_length[j + 1]
            if current[j] < best or (
                current[j] == best and current_length[j] < length
            ):
                best, length = current[j], current_length[j]
            if not np.isfinite(best):
                continue  # 带宽太窄时，某些格点没有可达的前驱。

            # 三项变量同时计算代价，共享同一条路径；横向加速度保留正负号。
            dv = a[i, 0] - b[j, 0]
            da_long = a[i, 1] - b[j, 1]
            da_lat = a[i, 2] - b[j, 2]
            local_cost = (dv * dv + da_long * da_long + da_lat * da_lat) / 3.0
            current[j + 1] = best + local_cost
            current_length[j + 1] = length + 1

        previous, current = current, previous
        previous_length, current_length = current_length, previous_length

    # 路径从两条序列的首点走到末点；不能跳过不易匹配的开头或结尾。
    return previous[m], previous_length[m]


def dtw_distance(a, b, band=0.1):
    """便于单独比较两辆车的接口；返回一个非负的无量纲距离。"""
    if not np.isfinite(band) or not 0 <= band <= 1:
        raise ValueError("DTW 带宽必须在 0～1 之间")
    arrays = [np.ascontiguousarray(x, dtype=np.float64) for x in (a, b)]
    for array in arrays:
        if array.ndim != 2 or array.shape[1] != 3 or len(array) < 2:
            raise ValueError("每条轨迹必须是至少两个时刻的 N×3 运动序列")
        if not np.isfinite(array).all():
            raise ValueError("运动序列包含 NaN 或无穷值")
    cost, length = cost_and_length(arrays[0], arrays[1], band)
    if not np.isfinite(cost) or length == 0:
        raise ValueError("当前带宽下没有首尾连通的 DTW 路径；请检查带宽和点数")
    return float(np.sqrt(cost / length))


@njit(cache=True, parallel=True)
def fill_distance_rows(values, offsets, begin, end, band, distances):
    """计算距离矩阵的一批上三角行，并同步填充对称位置。

    每对车辆只计算一次。prange 并行分配不同的 i，不会重复写同一格。
    offsets[i]:offsets[i+1] 是第 i 辆车在拼接数组中的记录范围。
    """
    n = len(offsets) - 1
    for i in prange(begin, end):
        a = values[offsets[i]:offsets[i + 1]]
        for j in range(i + 1, n):
            b = values[offsets[j]:offsets[j + 1]]
            cost, length = cost_and_length(a, b, band)
            distance = np.sqrt(cost / length) if length > 0 else np.inf
            distances[i, j] = distance
            distances[j, i] = distance
