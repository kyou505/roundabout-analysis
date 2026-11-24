"""从原始小汽车 CSV 生成可用于运动曲线比较的数据。

阅读顺序：先看文件末尾的 main()，再逐个查看对应函数。
只读取原 CSV 和两份 SQLite，不读取旧实验的样本名单或聚类标签。
质量检查只输出日志；真正供后续使用的序列和元信息写入 data/rdb2_processed。
"""

from collections import Counter
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data/rdb2_cars.csv"
TRAJECTORY_DB = ROOT / "data/OpenDD/rdb2/trajectories_rdb2_v3.sqlite"
MAP_FILE = ROOT / "data/OpenDD/rdb2/map_rdb2/map_rdb2.sqlite"
OUTPUT_DIR = ROOT / "data/rdb2_processed"

ID_COLUMNS = ["recording", "vehicle_id"]
MOTION_COLUMNS = ["speed_mps", "acc_lon_mps2", "acc_lat_mps2"]
# 每条变长轨迹是一个 N×7 数组；第 0 列为真实经过秒数。
SEQUENCE_COLUMNS = [
    "elapsed_s", "x_m", "y_m", "heading_rad", *MOTION_COLUMNS,
]
NUMERIC_COLUMNS = ["timestamp_s", *SEQUENCE_COLUMNS]
CHECK_COLUMNS = [
    "too_few_points", "too_few_times", "invalid_rows", "duplicate_times",
    "reversed_times", "large_gaps", "negative_speeds", "elapsed_mismatch",
]
CHECK_NAMES = [
    "少于3个记录", "有效时刻不足2个", "无效数值行数", "重复时刻多余行数",
    "时间倒序次数", "大间隔次数", "负速度记录数", "经过时间不一致行数",
]

# 这些参数复用当前实验的处理规则；改变参数属于新的实验设置。
MAX_GAP_S = 0.5
RECORDING_MARGIN_S = 0.04
SAMPLE_HZ = 10
SMOOTHING_SIGMA_S = 0.2
GATE_DISTANCE_M = 20.0
GATE_HALF_WIDTH_M = 6.0
MIN_HEADING_COSINE = 0.5       # 正向穿线时，车辆与道路方向夹角小于 60 度
MAP_DISTANCE_M = 3.0
MAP_HEADING_WEIGHT = 4.0
MIN_CONNECTION_S = 0.3
MIN_PHASE_S = 0.5
MIN_PHASE_DISTANCE_M = 0.1
SCALER_POINTS = 256

# 地图编号表示道路几何，不是行为标签。各链按行驶方向排列。
LANE_CHAINS = {
    "SE_in": [57, 58], "SW_in": [69, 70, 73],
    "NW_in": [85, 81], "NE_in": [90, 89],
    "SE_out": [61, 59, 55, 54], "SW_out": [77, 74, 71],
    "NW_out": [82, 83], "NE_out": [65, 86],
}
RING_LANES = {62, 63, 64, 66, 75, 76, 78, 79, 80, 84, 87, 88}
# 60 是连接道路，用于确定接近/连接/驶出三个区域；它本身不算环内道路。
CONNECTION_LANES = RING_LANES | {60}
ARM_BY_LANE = {
    **dict.fromkeys([54, 55, 56, 57, 58, 59, 61], "SE"),
    **dict.fromkeys([67, 68, 69, 70, 71, 72, 73, 74, 77], "SW"),
    **dict.fromkeys([81, 82, 83, 85], "NW"),
    **dict.fromkeys([65, 86, 89, 90], "NE"),
}


def load_trajectories(path=INPUT_FILE):
    """读取 CSV；身份保留为字符串，运动和时间字段转成数值。"""
    data = pd.read_csv(
        path, dtype={"recording": "string", "vehicle_id": "string"},
        # 往返解析保留导出 CSV 中浮点数的精度，尤其是较大的 UTM 坐标。
        float_precision="round_trip",
    )
    missing = set(ID_COLUMNS + NUMERIC_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"输入缺少字段：{sorted(missing)}")
    for column in ID_COLUMNS:
        data[column] = data[column].str.strip()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    print(f"[1] 读取 {len(data):,} 条时刻记录", flush=True)
    return data


def identity_is_valid(data):
    """身份缺失的行无法归入某一条轨迹；这里只生成掩码，不改原文件。"""
    return ~(data[ID_COLUMNS].isna().any(axis=1)
             | data[ID_COLUMNS].eq("").any(axis=1))


def check_quality(data):
    """每条轨迹生成一行检查结果；打印汇总，并把结果返回给筛选函数。"""
    valid_identity = identity_is_valid(data)
    print(f"[2] 身份缺失记录：{int((~valid_identity).sum()):,}")
    rows = []
    for (recording, vehicle_id), track in data.loc[valid_identity].groupby(
        ID_COLUMNS, sort=False
    ):
        values = track[NUMERIC_COLUMNS].to_numpy(dtype=float)
        times = track["timestamp_s"].to_numpy(dtype=float)
        finite_times = times[np.isfinite(times)]
        unique_times = np.unique(finite_times)
        intervals = np.diff(unique_times)
        speeds = track["speed_mps"].to_numpy(dtype=float)
        elapsed = track["elapsed_s"].to_numpy(dtype=float)
        comparable = np.isfinite(times) & np.isfinite(elapsed)
        mismatch = (
            np.sum(np.abs(elapsed[comparable]
                          - (times[comparable] - unique_times[0])) > 1e-6)
            if len(unique_times) else 0
        )
        rows.append({
            "recording": recording, "vehicle_id": vehicle_id,
            "n_points": len(track),
            "duration_s": float(np.ptp(unique_times)) if len(unique_times) else np.nan,
            "median_dt_s": float(np.median(intervals)) if len(intervals) else np.nan,
            "max_gap_s": float(intervals.max()) if len(intervals) else np.nan,
            "too_few_points": int(len(track) < 3),
            "too_few_times": int(len(unique_times) < 2),
            "invalid_rows": int((~np.isfinite(values).all(axis=1)).sum()),
            "duplicate_times": len(finite_times) - len(unique_times),
            "reversed_times": int((np.diff(finite_times) < 0).sum()),
            "large_gaps": int((intervals > MAX_GAP_S).sum()),
            "negative_speeds": int(((speeds < 0) & np.isfinite(speeds)).sum()),
            "elapsed_mismatch": int(mismatch),
        })
    quality = pd.DataFrame(rows)
    if quality.empty:
        raise ValueError("没有身份完整的轨迹")
    quality["passed"] = ~quality[CHECK_COLUMNS].gt(0).any(axis=1)
    print(f"    轨迹数 {len(quality):,}；基础检查通过 {int(quality.passed.sum()):,}")
    for column, name in zip(CHECK_COLUMNS, CHECK_NAMES):
        print(f"    {name}：涉及 {int(quality[column].gt(0).sum())} 条轨迹")
    for row in quality.loc[~quality.passed].head(20).to_dict("records"):
        reasons = [f"{name}={row[col]}" for col, name in zip(CHECK_COLUMNS, CHECK_NAMES)
                   if row[col] > 0]
        print(f"    检查样例 {row['recording']}:{row['vehicle_id']}：{'；'.join(reasons)}")
    return quality


def select_valid_trajectories(data, quality):
    """按整条轨迹筛选，不靠删去坏点把一段有缺口的轨迹拼成完整轨迹。"""
    keep = pd.MultiIndex.from_frame(quality.loc[quality.passed, ID_COLUMNS])
    identity = pd.MultiIndex.from_frame(data[ID_COLUMNS])
    selected = data.loc[identity.isin(keep)].copy()
    if selected.empty:
        raise ValueError("没有通过基础检查的轨迹")
    print(f"[3] 保留 {len(keep):,} 条轨迹、{len(selected):,} 条记录", flush=True)
    return selected


def load_map(path=MAP_FILE):
    """读取参考车道，并拆成短线段，供距离与朝向联合匹配。"""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        frame = pd.read_sql_query(
            "SELECT id, geometry FROM rdb2 WHERE type='trafficLane'", conn
        )
    lanes, lane_ids, starts, vectors = {}, [], [], []
    for row in frame.itertuples(index=False):
        text = row.geometry.split("(", 1)[1].rsplit(")", 1)[0]
        points = np.array([[float(v) for v in p.split()] for p in text.split(",")])
        lanes[int(row.id)] = points
        for start, end in zip(points[:-1], points[1:]):
            vector = end - start
            if np.dot(vector, vector) <= 0:
                raise ValueError(f"地图车道 {row.id} 包含零长度线段")
            lane_ids.append(int(row.id))
            starts.append(start)
            vectors.append(vector)
    required = RING_LANES | CONNECTION_LANES | set(ARM_BY_LANE)
    if not required.issubset(lanes):
        raise ValueError("当前地图缺少 rdb2 筛选规则所需的参考车道")
    vectors = np.array(vectors)
    print(f"[4] 读取 {len(lanes)} 条参考车道、{len(vectors)} 个线段")
    return {
        "lanes": lanes, "lane_ids": np.array(lane_ids),
        "starts": np.array(starts), "vectors": vectors,
        "squared_lengths": (vectors * vectors).sum(axis=1),
        "angles": np.arctan2(vectors[:, 1], vectors[:, 0]),
    }


def build_gates(road_map):
    """沿各方向参考车道，在距环内连接处 20 米的位置生成固定横断线。"""
    lanes, gates = road_map["lanes"], {}
    for name, chain in LANE_CHAINS.items():
        for left, right in zip(chain[:-1], chain[1:]):
            if np.linalg.norm(lanes[left][-1] - lanes[right][0]) >= 1e-5:
                raise ValueError(f"参考车道不连续：{name}，{left} → {right}")
        points = np.concatenate([
            lanes[lane_id] if i == 0 else lanes[lane_id][1:]
            for i, lane_id in enumerate(chain)
        ])
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        target = distances[-1] - GATE_DISTANCE_M if name.endswith("_in") else GATE_DISTANCE_M
        if not 0.5 < target < distances[-1] - 0.5:
            raise ValueError(f"参考车道长度不足：{name}")

        def point_at(distance):
            return np.array([np.interp(distance, distances, points[:, j]) for j in range(2)])

        center = point_at(target)
        forward = point_at(target + 0.5) - point_at(target - 0.5)
        forward /= np.linalg.norm(forward)
        tangent = np.array([-forward[1], forward[0]])
        gates[name] = {
            "name": name, "arm": name.split("_")[0], "kind": name.split("_")[1],
            "lane_chain": chain, "point": center.tolist(),
            "forward_normal": forward.tolist(), "line_tangent": tangent.tolist(),
            "segment_start": (center - GATE_HALF_WIDTH_M * tangent).tolist(),
            "segment_end": (center + GATE_HALF_WIDTH_M * tangent).tolist(),
            "ring_connection": (points[-1] if name.endswith("_in") else points[0]).tolist(),
            "reference_distance_m": GATE_DISTANCE_M, "half_width_m": GATE_HALF_WIDTH_M,
        }
    print(f"    生成 {len(gates)} 条固定观察边界")
    return gates


def load_recording_bounds(path=TRAJECTORY_DB):
    """从原数据库获取录像的观测起止时刻，不能仅用小汽车子集估计。"""
    bounds = {}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (recording,) in tables:
            if not recording.startswith("rdb2_"):
                continue
            table = recording.replace('"', '""')
            start, end = conn.execute(
                f'SELECT MIN(TIMESTAMP), MAX(TIMESTAMP) FROM "{table}"'
            ).fetchone()
            if start is None or end is None or end <= start:
                raise ValueError(f"录像时间范围无效：{recording}")
            bounds[recording] = (float(start), float(end))
    if not bounds:
        raise ValueError("没有找到 rdb2 录像")
    print(f"[5] 读取 {len(bounds)} 段录像的起止时刻")
    return bounds


def describe_route(sequence, road_map):
    """按地图判断接近、连接、驶出三个区域；这里只计算几何，不计算行为类别。"""
    source_t = sequence[:, 0]
    t = np.linspace(0., source_t[-1], int(np.ceil(source_t[-1] * SAMPLE_HZ)) + 1)
    xy = np.column_stack([np.interp(t, source_t, sequence[:, j]) for j in [1, 2]])
    heading = np.interp(t, source_t, np.unwrap(sequence[:, 3]))

    # 点投影到各参考线段，得到最近点距离。
    starts, vectors = road_map["starts"], road_map["vectors"]
    fraction = np.clip(
        ((xy[:, None, :] - starts) * vectors).sum(axis=2) / road_map["squared_lengths"],
        0., 1.,
    )
    distances = np.linalg.norm(
        xy[:, None, :] - (starts + fraction[:, :, None] * vectors), axis=2
    )
    # 同时考虑道路朝向，降低把车辆匹配到相邻反向道路的可能。
    score = distances ** 2 + MAP_HEADING_WEIGHT * (
        1 - np.cos(road_map["angles"] - heading[:, None])
    )
    nearest = score.argmin(axis=1)
    matched = road_map["lane_ids"][nearest]
    valid = distances[np.arange(len(t)), nearest] <= MAP_DISTANCE_M

    # 连续匹配连接道路至少 0.3 秒，才作为一次有效经过。
    connection = np.isin(matched, list(CONNECTION_LANES)) & valid
    changes = np.diff(np.r_[False, connection[:-1], False].astype(int))
    runs = [
        (a, b) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
        if t[b] - t[a] >= MIN_CONNECTION_S
    ]
    if not runs:
        return None
    enter, leave = t[runs[0][0]], t[runs[-1][1]]
    phase_bounds = np.array([0., enter, leave, t[-1]])
    if np.any(np.diff(phase_bounds) < MIN_PHASE_S):
        return None
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if np.any(np.diff(np.interp(phase_bounds, t, travel)) < MIN_PHASE_DISTANCE_M):
        return None

    # 复用原流程：首尾一秒内的最优道路匹配投票确定方向；
    # 环内/连接道路的识别另受上面的 3 米有效距离约束。
    def majority_arm(mask):
        arms = [ARM_BY_LANE[int(lane)] for lane in matched[mask] if int(lane) in ARM_BY_LANE]
        # 并列时按名称排序，避免 set 的遍历顺序造成不可复现。
        return max(sorted(set(arms)), key=arms.count) if arms else "unknown"

    origin = majority_arm(t <= min(1., t[-1]))
    destination = majority_arm(t >= max(0., t[-1] - 1.))
    ring_seconds = float(
        (np.isin(matched, list(RING_LANES)) & valid)[:-1].sum() * (t[1] - t[0])
    )
    return {
        "route": f"{origin}>{destination}", "origin": origin, "destination": destination,
        "enter_s": float(enter), "leave_s": float(leave),
        "ring_seconds": ring_seconds, "unknown_fraction": float(1 - valid.mean()),
    }


def find_gate_crossings(sequence, gate):
    """检测有限横断线上的正向穿越，线性插值计算实际穿线时刻。"""
    center = np.array(gate["point"])
    forward = np.array(gate["forward_normal"])
    tangent = np.array(gate["line_tangent"])
    # 负值在观察线之前，正值在之后；符号由道路行驶方向确定。
    signed_distance = ((sequence[:, 1:3] - center) * forward).sum(axis=1)
    candidates = np.flatnonzero(
        (signed_distance[:-1] <= 0) & (signed_distance[1:] > 0)
    )
    heading = np.unwrap(sequence[:, 3])
    crossings = []
    for index in candidates:
        fraction = -signed_distance[index] / (
            signed_distance[index + 1] - signed_distance[index]
        )
        xy = sequence[index, 1:3] + fraction * (
            sequence[index + 1, 1:3] - sequence[index, 1:3]
        )
        lateral = float(np.dot(xy - center, tangent))
        angle = heading[index] + fraction * (heading[index + 1] - heading[index])
        cosine = float(np.dot([np.cos(angle), np.sin(angle)], forward))
        if abs(lateral) <= GATE_HALF_WIDTH_M and cosine > MIN_HEADING_COSINE:
            time = sequence[index, 0] + fraction * (
                sequence[index + 1, 0] - sequence[index, 0]
            )
            crossings.append({
                "time_s": float(time), "lateral_m": lateral, "heading_cosine": cosine,
            })
    return crossings


def crop_sequence(sequence, start, end):
    """只插值两个边界点；中间原始位置、速度和加速度逐值保留。"""
    def interpolate(time):
        row = np.array([
            np.interp(time, sequence[:, 0], sequence[:, j])
            for j in range(sequence.shape[1])
        ])
        # 朝向跨越 ±π 时先展开，防止直接插值得到相反方向。
        row[3] = np.interp(time, sequence[:, 0], np.unwrap(sequence[:, 3]))
        return row

    inside = sequence[(sequence[:, 0] > start) & (sequence[:, 0] < end)]
    cropped = np.vstack([interpolate(start), inside, interpolate(end)])
    cropped[:, 0] -= start
    if not np.all(np.diff(cropped[:, 0]) > 0):
        raise ValueError("截取后出现重复或倒序时刻")
    if not np.array_equal(cropped[1:-1, 1:], inside[:, 1:]):
        raise AssertionError("截取改变了内部原始观测值")
    return cropped


def select_and_crop_passages(data, road_map, gates, recording_bounds):
    """从基础有效轨迹中选择完整环岛通行，并截取对应入口线至出口线。"""
    missing = set(data.recording.unique()) - set(recording_bounds)
    if missing:
        raise ValueError(f"找不到录像时间范围：{sorted(missing)}")
    passages, rows = [], []
    excluded, examples = Counter(), {}
    complete_count = ring_count = input_count = 0
    max_endpoint_error = 0.

    def reject(reason, key):
        excluded[reason] += 1
        examples.setdefault(reason, [])
        if len(examples[reason]) < 3:
            examples[reason].append(key)

    print("[6] 开始地图筛选和固定边界截取", flush=True)
    for recording, frame in data.groupby("recording", sort=True):
        before = len(passages)
        recording_start, recording_end = recording_bounds[recording]
        for vehicle_id, track in frame.groupby("vehicle_id", sort=True):
            input_count += 1
            key = f"{recording}:{vehicle_id}"
            # 基础质检已验证时间顺序，这里沿用原始记录的顺序。
            sequence = track[
                ["timestamp_s", "x_m", "y_m", "heading_rad", *MOTION_COLUMNS]
            ].to_numpy(dtype=float, copy=True)
            raw_start, raw_end = float(sequence[0, 0]), float(sequence[-1, 0])
            if (raw_start <= recording_start + RECORDING_MARGIN_S
                    or raw_end >= recording_end - RECORDING_MARGIN_S):
                reject("recording_boundary", key)
                continue
            sequence[:, 0] -= raw_start
            geometry = describe_route(sequence, road_map)
            if geometry is None:
                reject("incomplete_three_phases", key)
                continue
            complete_count += 1
            if geometry["ring_seconds"] < MIN_CONNECTION_S:
                reject("bypass", key)
                continue
            ring_count += 1
            start_gate = geometry["origin"] + "_in"
            end_gate = geometry["destination"] + "_out"
            if start_gate not in gates or end_gate not in gates:
                reject("unknown_route", key)
                continue

            starts = find_gate_crossings(sequence, gates[start_gate])
            ends = find_gate_crossings(sequence, gates[end_gate])
            if not starts:
                reject("missing_start_crossing", key)
                continue
            if not ends:
                reject("missing_end_crossing", key)
                continue
            # 首次正向入口穿越到最后正向出口穿越，保留内部绕行和停留。
            first, last = starts[0], ends[-1]
            start, end = first["time_s"], last["time_s"]
            if end <= start:
                reject("invalid_crossing_order", key)
                continue
            cropped = crop_sequence(sequence, start, end)
            for point, gate_name in [(cropped[0, 1:3], start_gate), (cropped[-1, 1:3], end_gate)]:
                gate = gates[gate_name]
                error = abs(float(np.dot(
                    point - np.array(gate["point"]), np.array(gate["forward_normal"])
                )))
                max_endpoint_error = max(max_endpoint_error, error)
                if error > 1e-6:
                    raise AssertionError(f"{key} 的边界点没有落在横断线上")

            passages.append(cropped)
            rows.append({
                "key": key, "recording": recording, "vehicle_id": vehicle_id,
                "route": geometry["route"], "start_gate": start_gate, "end_gate": end_gate,
                "source_duration_s": float(sequence[-1, 0]), "source_raw_start_s": raw_start,
                "crop_start_source_elapsed_s": start, "crop_end_source_elapsed_s": end,
                # 保留原录像时刻，后续才能关联同一时刻的其他车辆。
                "raw_start_s": raw_start + start, "raw_end_s": raw_start + end,
                "duration_s": end - start,
                "path_length_m": float(np.linalg.norm(np.diff(cropped[:, 1:3], axis=0), axis=1).sum()),
                "n_raw": len(cropped),
                "n_start_crossings": len(starts), "n_end_crossings": len(ends),
                "start_lateral_m": first["lateral_m"], "end_lateral_m": last["lateral_m"],
                "start_heading_cosine": first["heading_cosine"],
                "end_heading_cosine": last["heading_cosine"],
            })
        print(f"    {recording}：保留 {len(passages) - before} 条", flush=True)

    if not passages:
        raise ValueError("没有满足当前通行范围要求的轨迹")
    metadata = pd.DataFrame(rows)
    if metadata.key.duplicated().any() or input_count != len(metadata) + sum(excluded.values()):
        raise AssertionError("轨迹身份重复或筛选数量不守恒")
    # attrs 是附着在表上的汇总信息，不是逐车数据列，也不另存质检 CSV。
    metadata.attrs["selection_summary"] = {
        "input_trajectories": input_count, "complete_three_phase_passages": complete_count,
        "ring_before_gate_crop": ring_count, "accepted": len(metadata),
        "excluded_by_reason": dict(excluded), "excluded_examples": examples,
        "max_endpoint_line_error_m": max_endpoint_error,
        "retained_original_values_unchanged": True,
    }
    print(f"    三阶段完整 {complete_count:,}；经过环岛 {ring_count:,}；成功截取 {len(metadata):,}")
    for reason, count in excluded.items():
        print(f"    排除 {reason}：{count}；样例：{', '.join(examples[reason])}")
    return passages, metadata


def prepare_motion_curves(passages):
    """约 10 Hz 重采样；只平滑三项运动信号，保留真实秒数和共同位置坐标。"""
    curves = []
    for sequence in passages:
        duration = sequence[-1, 0]
        # 包含两端点，实际间隔 <= 0.1 秒；不同车辆仍有不同点数和总时长。
        t = np.linspace(0., duration, int(np.ceil(duration * SAMPLE_HZ)) + 1)
        curve = np.column_stack([
            t, *[np.interp(t, sequence[:, 0], sequence[:, j]) for j in range(1, 7)],
        ])
        curve[:, 3] = np.interp(t, sequence[:, 0], np.unwrap(sequence[:, 3]))
        # sigma 的单位原本是秒，除以采样间隔后变成高斯滤波所需的点数。
        curve[:, 4:7] = gaussian_filter1d(
            curve[:, 4:7], SMOOTHING_SIGMA_S / (t[1] - t[0]), axis=0, mode="nearest"
        )
        if not np.isfinite(curve).all() or not np.all(np.diff(curve[:, 0]) > 0):
            raise AssertionError("曲线处理后产生无效数值或非递增时间")
        curves.append(curve)
    print(f"[7] 完成 {len(curves):,} 条变长曲线的重采样与平滑")
    return curves


def assign_recording_split(metadata, recordings):
    """按全部原始录像名称排序，每第 4 段用于内部复核；同一录像不拆开。"""
    folds = {recording: index % 4 for index, recording in enumerate(sorted(recordings))}
    result = metadata.copy()
    result["fold"] = result.recording.map(folds)
    if result.fold.isna().any():
        raise ValueError("样本包含未知录像")
    result["split"] = np.where(result.fold.eq(3), "internal_review", "discovery")
    # 这些录像已参与过探索，因此叫“内部复核”，不称为独立测试集。
    print(f"[8] 录像划分：{result['split'].value_counts().to_dict()}")
    return result


def standardize_curves(curves, metadata):
    """只用发现集拟合三个通道的共同尺度，再应用到所有车辆。"""
    fit = metadata.split.eq("discovery").to_numpy()
    if len(curves) != len(metadata) or not fit.any():
        raise ValueError("曲线与元信息未对齐，或没有发现集样本")
    # 为复现原实验，每条曲线取 256 个等经过时间比例点来估计总体尺度。
    # 这样每辆车等权，不让观测更久、点数更多的车辆主导均值。
    # 这里的 256 点仅用于尺度估计；后面保存和比较的仍是变长序列。
    fixed = np.array([
        np.column_stack([
            np.interp(np.linspace(0., curve[-1, 0], SCALER_POINTS), curve[:, 0], curve[:, j])
            for j in range(4, 7)
        ])
        for curve in curves
    ])
    mean = fixed[fit].mean(axis=(0, 1))
    std = fixed[fit].std(axis=(0, 1))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("发现集运动通道没有有效的非零尺度")
    scaled_curves = [(curve[:, 4:7] - mean) / std for curve in curves]
    scaler = {
        "channels": MOTION_COLUMNS, "units": ["m/s", "m/s^2", "m/s^2"],
        "mean": mean.tolist(), "std": std.tolist(), "fit_split": "discovery",
        "fit_trajectories": int(fit.sum()), "points_per_trajectory_for_fit": SCALER_POINTS,
        "weighting": "equal trajectories, equal elapsed-time fractions within each",
        "output": "variable-length, three motion channels only",
    }
    print("    三通道共同均值：", np.round(mean, 6))
    print("    三通道共同标准差：", np.round(std, 6))
    return scaled_curves, scaler


def save_prepared_data(curves, scaled_curves, metadata, scaler, passages, gates,
                       quality, recording_bounds, output_dir=OUTPUT_DIR):
    """保存后续真正需要的数组和身份信息；检查过程本身只输出日志。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = metadata.key.to_numpy(dtype=str)
    for arrays in [passages, curves, scaled_curves]:
        if len(arrays) != len(keys):
            raise AssertionError("轨迹数组和身份数量不一致")

    def save_sequences(filename, arrays, columns):
        # 拼接所有车辆的行，用 offsets 标出每辆车的开始与结束位置；
        # 读取第 i 辆车：values[offsets[i]:offsets[i + 1]]。
        offsets = np.r_[0, np.cumsum([len(array) for array in arrays])].astype("int64")
        values = np.concatenate(arrays)
        if values.shape[1] != len(columns) or not np.isfinite(values).all():
            raise AssertionError(f"{filename} 的列数或数值不合法")
        np.savez_compressed(
            output_dir / filename, keys=keys, values=values, offsets=offsets,
            columns=np.array(columns, dtype=str),
        )

    save_sequences("raw_sequences.npz", passages, SEQUENCE_COLUMNS)
    save_sequences("clean_sequences.npz", curves, SEQUENCE_COLUMNS)
    save_sequences("motion_sequences.npz", scaled_curves, MOTION_COLUMNS)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    summary = {
        "raw_input": str(INPUT_FILE), "map_input": str(MAP_FILE),
        "recording_bounds_input": str(TRAJECTORY_DB),
        "quality_trajectories": len(quality),
        "quality_passed": int(quality.passed.sum()),
        "selection": metadata.attrs["selection_summary"],
        "split_counts": {k: int(v) for k, v in metadata.split.value_counts().items()},
        "recordings_per_split": {
            k: int(v) for k, v in metadata.groupby("split").recording.nunique().items()
        },
        "recording_order": sorted(recording_bounds),
        "parameters": {
            "sample_hz": SAMPLE_HZ, "smoothing_sigma_s": SMOOTHING_SIGMA_S,
            "gate_distance_m": GATE_DISTANCE_M, "gate_half_width_m": GATE_HALF_WIDTH_M,
            "minimum_heading_cosine": MIN_HEADING_COSINE,
            "map_distance_m": MAP_DISTANCE_M, "map_heading_weight": MAP_HEADING_WEIGHT,
            "max_gap_s": MAX_GAP_S, "recording_margin_s": RECORDING_MARGIN_S,
            "minimum_connection_s": MIN_CONNECTION_S, "minimum_phase_s": MIN_PHASE_S,
            "minimum_phase_distance_m": MIN_PHASE_DISTANCE_M,
        },
        "duration_s": {
            "min": float(metadata.duration_s.min()), "median": float(metadata.duration_s.median()),
            "max": float(metadata.duration_s.max()),
        },
        "motion_input_has_position_or_time_channels": False,
        "no_clustering_performed": True,
    }
    for name, obj in [("gates.json", gates), ("scaler.json", scaler), ("summary.json", summary)]:
        (output_dir / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"[9] 已保存至：{output_dir}", flush=True)
    print("    后续距离计算读取 motion_sequences.npz；位置、实际时间和身份另行保留。")


def main():
    """唯一运行入口：每一步的返回值，直接传给下一步。"""
    # 1～3：读取原始表，检查质量，再按整条轨迹筛选。
    raw_data = load_trajectories()
    quality = check_quality(raw_data)
    valid_data = select_valid_trajectories(raw_data, quality)
    del raw_data  # 后续不再使用这一份内存副本，原 CSV 不变。

    # 4～6：读取地图与录像范围，识别完整环岛通行，截取固定边界之间的记录。
    road_map = load_map()
    gates = build_gates(road_map)
    recording_bounds = load_recording_bounds()
    passages, metadata = select_and_crop_passages(
        valid_data, road_map, gates, recording_bounds
    )
    del valid_data

    # 7～8：处理曲线；划分整段录像；只在发现部分拟合共同标准化参数。
    curves = prepare_motion_curves(passages)
    metadata["n_clean"] = [len(curve) for curve in curves]
    metadata = assign_recording_split(metadata, recording_bounds)
    scaled_curves, scaler = standardize_curves(curves, metadata)

    # 9：仅保存后续分析会使用的数据，不写逐车质检 CSV。
    save_prepared_data(
        curves, scaled_curves, metadata, scaler, passages, gates,
        quality, recording_bounds,
    )


if __name__ == "__main__":
    main()
