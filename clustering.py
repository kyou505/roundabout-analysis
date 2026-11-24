"""接续 data_process.py：运动序列 → DTW 距离 → 平均链接聚类。

从文件末尾 main() 阅读全流程。输入只来自 data/rdb2_processed，
不读取历史实验距离或标签；输出写入已由 Git 忽略的 experiments/。
默认枚举 K=2～12，不自动把最高轮廓系数对应的 K 当成最终行为类别数。
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numba import config as numba_config, set_num_threads
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples
from threadpoolctl import threadpool_limits

import dtw


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data/rdb2_processed"
OUTPUT_DIR = ROOT / "experiments/rdb2_dtw_clustering"
CHANNELS = ["speed_mps", "acc_lon_mps2", "acc_lat_mps2"]
INPUT_FILES = ["motion_sequences.npz", "metadata.csv", "scaler.json"]


def file_sha256(path):
    """记录输入内容，防止数据已变而误用旧距离矩阵。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(input_dir):
    return {name: file_sha256(input_dir / name) for name in INPUT_FILES}


def write_json(path, content):
    """先完整写入临时文件再替换，避免中断后进度文件只剩半截。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_prepared_data(input_dir):
    """读取变长序列，并核对矩阵行序、轨迹身份、录像划分和共同尺度。"""
    metadata = pd.read_csv(
        input_dir / "metadata.csv",
        dtype={"key": str, "recording": str, "vehicle_id": str},
    )
    required = {"key", "recording", "vehicle_id", "split", "n_clean"}
    if not required.issubset(metadata.columns) or metadata[list(required)].isna().any().any():
        raise ValueError("metadata.csv 缺少必要字段或存在空值")
    with np.load(input_dir / "motion_sequences.npz", allow_pickle=False) as data:
        keys = data["keys"]
        offsets = data["offsets"]
        values = np.ascontiguousarray(data["values"], dtype=np.float64)
        columns = data["columns"].tolist()
    n = len(metadata)
    if columns != CHANNELS or values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("输入必须是按约定顺序排列的三通道运动序列")
    if not np.array_equal(keys, metadata.key.to_numpy()) or metadata.key.duplicated().any():
        raise ValueError("数组与 metadata 的车辆身份/顺序不一致，或身份重复")
    if (offsets.shape != (n + 1,) or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != len(values)
            or np.any(np.diff(offsets) < 2)):
        raise ValueError("变长序列 offsets 非法，或某条轨迹不足两个时刻")
    if not np.array_equal(np.diff(offsets), metadata.n_clean.to_numpy()):
        raise ValueError("metadata 中的点数与实际运动序列不一致")
    if not np.isfinite(values).all():
        raise ValueError("运动序列存在 NaN 或无穷值")
    if not metadata.split.isin(["discovery", "internal_review"]).all():
        raise ValueError("存在未知的数据划分")
    if metadata.groupby("recording").split.nunique().max() > 1:
        raise ValueError("同一录像被拆到发现集与内部复核集")
    fit_count = int(metadata.split.eq("discovery").sum())
    if fit_count < 3:
        raise ValueError("至少需要三条发现集轨迹才能比较聚类切分")
    scaler = json.loads((input_dir / "scaler.json").read_text(encoding="utf-8"))
    if (scaler["channels"] != CHANNELS or scaler["fit_split"] != "discovery"
            or scaler["fit_trajectories"] != fit_count):
        raise ValueError("标准化参数与当前发现集不一致，请重新运行预处理")
    # 输入已经标准化；这里不再逐车减均值，也不重新拟合尺度。
    print(f"[1] 读取 {n:,} 条轨迹、{len(values):,} 个时刻；发现集 {fit_count:,} 条", flush=True)
    return metadata, values, np.ascontiguousarray(offsets, dtype=np.int64)


def validate_distances(distances, n):
    """按块检查矩阵，避免另建一个同样大小的布尔矩阵。"""
    if distances.shape != (n, n) or not np.all(np.diag(distances) == 0):
        raise ValueError("距离矩阵大小不符或对角线不为零")
    for begin in range(0, n, 256):
        end = min(begin + 256, n)
        block = distances[begin:end]
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError("距离存在负数/非有限值；可能是带宽下无可行路径")
        if not np.array_equal(block, distances[:, begin:end].T):
            raise ValueError("距离矩阵不对称")


def calculate_distances(values, offsets, source_hashes, output_dir, band, block_rows):
    """分块计算全体车辆的距离，落盘后记录进度；同配置再次运行可续算。"""
    n = len(offsets) - 1
    config = {
        "n": n, "channels": CHANNELS, "source_hashes": source_hashes,
        "dtw_code_sha256": file_sha256(Path(dtw.__file__)),
        "band_fraction": band,
        "band_definition": "abs(i/(n-1) - j/(m-1)) <= band_fraction",
        "local_cost": "mean squared difference of three jointly standardized channels",
        "distance": "sqrt(minimum cumulative cost / selected path length)",
        "tie_break": "shortest path among equal cumulative costs",
        "shared_alignment_path": True, "matrix_dtype": "float32",
        "duration_difference_is_separate_feature": False,
        "normalization_note": "post hoc path normalization; not minimum mean-cost path",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "distance_config.json"
    progress_path = output_dir / "distance_progress.json"
    matrix_path = output_dir / "distances.npy"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("输入、DTW 实现或带宽已变化；请用 --output-dir 指定新的输出目录")
    elif matrix_path.exists() or progress_path.exists():
        raise ValueError("目录存在缺少配置的距离文件，请使用新的输出目录")
    else:
        write_json(config_path, config)

    begin = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        begin = progress["completed_rows"]
        if progress["n"] != n or not isinstance(begin, int) or not 0 <= begin <= n:
            raise ValueError("距离进度记录与当前数据不一致")
        distances = np.load(matrix_path, mmap_mode="r+")
        if distances.shape != (n, n) or distances.dtype != np.dtype("float32"):
            raise ValueError("已有距离文件的大小或类型不符")
    else:
        distances = np.lib.format.open_memmap(
            matrix_path, mode="w+", dtype="float32", shape=(n, n),
        )
        distances[:] = 0
        distances.flush()
        write_json(progress_path, {"n": n, "completed_rows": 0})

    started = time.perf_counter()
    if begin == n:
        print("[2] 输入和参数未变，复用已完成的 DTW 距离矩阵", flush=True)
    else:
        print(f"[2] 计算 {n * (n - 1) // 2:,} 对 DTW 距离，从第 {begin} 行继续", flush=True)
        for first in range(begin, n, block_rows):
            end = min(first + block_rows, n)
            dtw.fill_distance_rows(values, offsets, first, end, band, distances)
            if not np.isfinite(distances[first:end]).all():
                raise ValueError("某些序列在当前带宽下无法对齐；请增大带宽并使用新的输出目录")
            # 先确保矩阵已落盘，再推进完成行数；中断时最多重算当前块。
            distances.flush()
            write_json(progress_path, {"n": n, "completed_rows": end})
            percent = end * (2 * n - end - 1) / (n * (n - 1)) * 100
            print(f"    完成 {percent:5.1f}% 的车辆对，累计 {time.perf_counter() - started:.1f} 秒", flush=True)
    validate_distances(distances, n)
    print(f"    距离矩阵通过检查：{n}×{n}，对称、非负、有限、对角线为零", flush=True)
    return distances


def build_discovery_tree(distances, metadata):
    """只用发现集建立平均链接树；复核集不参与确定树结构。"""
    fit_indices = np.flatnonzero(metadata.split.eq("discovery").to_numpy())
    fit_distances = np.asarray(distances[np.ix_(fit_indices, fit_indices)], dtype=np.float64)
    # linkage 的二维输入被解释为“特征”，不能把 N×N 距离矩阵直接传入。
    # squareform 把上三角压成一维距离列表，显式告诉它使用已有的 DTW 距离。
    condensed = squareform(fit_distances, checks=False)
    tree = linkage(condensed, method="average")
    # 平均链接：两簇间距离 = 两簇所有跨簇车辆对距离的平均值。
    print(f"[3] 用 {len(fit_indices):,} 条发现集轨迹建立平均链接聚类树", flush=True)
    return fit_indices, fit_distances, tree


def relabel_by_size(labels):
    """从大到小编号为 C0、C1……；编号本身不包含行为含义。"""
    counts = pd.Series(labels).value_counts()
    groups = sorted(counts.index, key=lambda group: (-int(counts[group]), int(group)))
    mapping = {group: rank for rank, group in enumerate(groups)}
    return np.array([mapping[label] for label in labels], dtype=np.int64)


def evaluate_cuts(distances, metadata, fit_indices, fit_distances, tree, k_values):
    """枚举切分，保存每辆车的归组、轮廓系数和可回看的代表车辆。"""
    review_indices = np.flatnonzero(metadata.split.eq("internal_review").to_numpy())
    assignments = metadata.copy()
    candidates, cluster_rows = [], []
    for requested_k in k_values:
        # maxclust 表示至多 K 簇；合并高度相同时，实际簇数可能小于请求值。
        fit_labels = relabel_by_size(fcluster(tree, requested_k, criterion="maxclust"))
        actual_k = len(np.unique(fit_labels))
        counts = np.bincount(fit_labels)
        # 这里使用已经计算好的 DTW 距离，不重新对距离矩阵做欧氏距离。
        silhouette = (
            silhouette_samples(fit_distances, fit_labels, metric="precomputed")
            if 1 < actual_k < len(fit_indices) else np.full(len(fit_indices), np.nan)
        )
        all_labels = np.empty(len(metadata), dtype=np.int64)
        all_labels[fit_indices] = fit_labels
        review_means = np.empty((len(review_indices), actual_k))
        local_members = []
        for cluster in range(actual_k):
            local = np.flatnonzero(fit_labels == cluster)
            local_members.append(local)
            if len(review_indices):
                # 层次聚类没有天然 predict。此处明确采用“到发现簇的平均距离”归组。
                review_means[:, cluster] = distances[
                    np.ix_(review_indices, fit_indices[local])
                ].mean(axis=1, dtype=np.float64)
        if len(review_indices):
            all_labels[review_indices] = review_means.argmin(axis=1)
        assignments[f"cluster_k{requested_k}"] = all_labels
        review_counts = np.bincount(all_labels[review_indices], minlength=actual_k)
        row = {
            "requested_k": requested_k, "actual_k": actual_k,
            "silhouette": float(silhouette.mean()) if np.isfinite(silhouette).all() else None,
            "negative_silhouette_fraction": float(np.mean(silhouette < 0)) if np.isfinite(silhouette).all() else None,
            "largest_cluster_fraction": float(counts.max() / len(fit_indices)),
            "smallest_cluster_n": int(counts.min()),
            "discovery_counts": json.dumps(counts.tolist()),
            "internal_review_counts": json.dumps(review_counts.tolist()),
        }
        candidates.append(row)
        for cluster, local in enumerate(local_members):
            # medoid 是到本簇其他车辆平均距离最小的真实车辆，不是虚构的均值曲线。
            within = fit_distances[np.ix_(local, local)]
            medoid_local = local[within.mean(axis=1).argmin()]
            medoid_global = fit_indices[medoid_local]
            cluster_rows.append({
                "requested_k": requested_k, "actual_k": actual_k, "cluster": cluster,
                "discovery_n": int(counts[cluster]), "internal_review_n": int(review_counts[cluster]),
                "discovery_silhouette": float(silhouette[local].mean()) if np.isfinite(silhouette[local]).all() else None,
                "medoid_key": metadata.iloc[medoid_global]["key"],
                "medoid_global_index": int(medoid_global),
            })
        score = f"{row['silhouette']:.4f}" if row["silhouette"] is not None else "不可计算"
        print(f"    K={requested_k:2d}，实际 {actual_k} 簇，轮廓={score}，发现集数量={counts.tolist()}", flush=True)
    return assignments, pd.DataFrame(candidates), pd.DataFrame(cluster_rows)


def save_clustering_results(output_dir, metadata, tree, fit_indices, assignments, candidates,
                            cluster_summary, k_values, threads):
    """保存树、候选分组和结果说明；不把候选簇自动命名为驾驶行为。"""
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    np.save(output_dir / "linkage.npy", tree)
    np.save(output_dir / "discovery_indices.npy", fit_indices)
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    candidates.to_csv(output_dir / "k_candidates.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    write_json(output_dir / "clustering_summary.json", {
        "n": len(metadata), "discovery_n": len(fit_indices),
        "internal_review_n": int(metadata.split.eq("internal_review").sum()),
        "channels": CHANNELS, "linkage": "average", "fit_split": "discovery",
        "k_candidates": k_values, "final_k_selected": None,
        "review_assignment": "minimum mean DTW distance to discovery cluster",
        "review_assignment_note": "assignment only; not a clustering stability test",
        "cluster_numbering": "descending discovery size; no behavior names assigned",
        "silhouette_scope": "all discovery samples; precomputed DTW distance",
        "threads": threads, "source_unchanged": True,
        "distance_matrix_order": "metadata.csv row order; same as motion_sequences keys",
        "tree_leaf_order": "metadata.iloc[discovery_indices.npy]",
    })
    print(f"[5] 输出已保存：{output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="预处理输出目录")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="本轮距离和聚类结果目录")
    parser.add_argument("--band", type=float, default=0.1, help="DTW 相对进度带宽，默认 0.1")
    parser.add_argument("--threads", type=int, default=min(4, numba_config.NUMBA_NUM_THREADS), help="Numba 计算线程数")
    parser.add_argument("--block-rows", type=int, default=128, help="每完成多少行保存一次距离进度")
    parser.add_argument("--k-min", type=int, default=2, help="候选最小簇数")
    parser.add_argument("--k-max", type=int, default=12, help="候选最大簇数")
    args = parser.parse_args()
    if not np.isfinite(args.band) or not 0 <= args.band <= 1:
        parser.error("--band 必须在 0～1 之间")
    if not 1 <= args.threads <= numba_config.NUMBA_NUM_THREADS or args.block_rows < 1:
        parser.error("线程数须在本机可用范围内，block-rows 须为正整数")
    if not 2 <= args.k_min <= args.k_max:
        parser.error("候选簇数必须满足 2 <= k-min <= k-max")
    return args


def main():
    args = parse_args()
    set_num_threads(args.threads)
    source_hashes = input_hashes(args.input_dir)
    metadata, values, offsets = load_prepared_data(args.input_dir)
    if args.k_max >= int(metadata.split.eq("discovery").sum()):
        raise ValueError("候选最大簇数必须小于发现集轨迹数")
    k_values = list(range(args.k_min, args.k_max + 1))

    # 1. 计算全体车辆两两距离。复核集参与距离计算，但不参与建树或拟合尺度。
    distances = calculate_distances(
        values, offsets, source_hashes, args.output_dir, args.band, args.block_rows,
    )
    with threadpool_limits(limits=1):
        # 2. 从全矩阵中取发现集子矩阵，建立一棵平均链接聚类树。
        fit_indices, fit_distances, tree = build_discovery_tree(distances, metadata)
        # 3. 切成不同数量的簇，比较分离程度与簇规模，并保存复核集归组。
        print("[4] 比较候选切分，保留小簇，不自动选择最终 K", flush=True)
        assignments, candidates, cluster_summary = evaluate_cuts(
            distances, metadata, fit_indices, fit_distances, tree, k_values,
        )
    if input_hashes(args.input_dir) != source_hashes:
        raise RuntimeError("运行期间输入被修改，本轮结果不能作为同一输入的结果保存")
    # 4. 每个输出都保留车辆顺序，便于下一步关联原始曲线与交通背景。
    save_clustering_results(
        args.output_dir, metadata, tree, fit_indices, assignments, candidates,
        cluster_summary, k_values, args.threads,
    )


if __name__ == "__main__":
    main()
