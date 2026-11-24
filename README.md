# 环岛车辆通行轨迹聚类

以一辆小汽车在 OpenDD rdb2 环岛的一次完整通行为样本。使用速度、纵向加速度和横向加速度三条曲线计算多变量 DTW 距离，再用平均链接层次聚类比较不同簇数下的分组。

本仓库提交的是 **rdb2 数据处理 → DTW → 平均链接层次聚类**的核心流程。`experiments/` 是本地探索与输出目录，不纳入版本管理；其中的谱聚类、rdb3/rdb4 对照及报告不属于本版本的复现范围。

## 运行

从 [OpenDD 数据集主页](https://l3pilot.eu/data/opendd.html) 获取 rdb2，将数据包中的两个原始文件放在以下位置。数据来源、许可和版本校验见 [data/README.md](data/README.md)。数据引用见 Breuer 等（2020），*OpenDD: A Large-Scale Roundabout Drone Dataset*。

```text
data/OpenDD/rdb2/trajectories_rdb2_v3.sqlite
data/OpenDD/rdb2/map_rdb2/map_rdb2.sqlite
```

在项目根目录使用 Python 3.10 运行（本次实验为 3.10.13）。

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python data_export.py   # 从各段录像提取小汽车轨迹，写入 data/rdb2_cars.csv
.venv/bin/python data_process.py  # 检查轨迹质量，截取固定边界间的完整通行，生成变长运动序列
.venv/bin/python clustering.py    # 调用 dtw.py 计算距离，建立聚类树并输出候选分组
```

预处理按约 10 Hz 重采样并平滑运动曲线，按录像划分发现集和复核集。三个通道统一使用发现集估计的尺度。已有 `data/rdb2_processed/` 的完整输出时，可直接运行第三个脚本。

## 查看结果

聚类结果位于 `experiments/rdb2_dtw_clustering/`。默认在发现集上比较 K=2～12，复核车辆归入平均 DTW 距离最小的发现簇。

| 文件 | 用途 |
| --- | --- |
| `k_candidates.csv` | 比较各候选 K 的轮廓系数和簇规模 |
| `assignments.csv` | 按录像和车辆 ID 查询每条轨迹在各个 K 下的簇编号 |
| `cluster_summary.csv` | 查找每簇的样本数和代表车辆，通过其索引回看 `data/rdb2_processed/clean_sequences.npz` 中的运动曲线 |

相同输入和带宽下可复用距离矩阵。聚类参数见 `.venv/bin/python clustering.py --help`。可选的 `map.py` 在导出 CSV 后运行，用于查看一辆示例车的轨迹与车道参考线。

## 复现核对

1. 查看 `data/rdb2_processed/summary.json`。全量 rdb2 得到 **5,886** 条完整通行轨迹，发现集 4,381 条（42 段录像），内部复核集 1,505 条（14 段录像）。
2. 查看 `k_candidates.csv` 的 `requested_k=7` 行。发现集各簇规模从大到小为 **2,295、1,041、1,012、29、2、1、1**，轮廓系数约 **0.3752**。研究选择这行分析三个主要群体。`K=2` 的轮廓系数较高（0.4683），对应的切分是 4,379 和 2 条。

数据文件不会随 Git 克隆下载。运行前按 [数据说明](data/README.md) 下载，并核对 [SHA256SUMS](data/SHA256SUMS) 中的版本。

## 版本管理范围

- 提交：根目录 Python 代码、`requirements.txt`、README、`.gitignore`，以及 `data/` 中的下载说明和校验清单。
- 本地保留：原始 SQLite、导出 CSV、处理后的 NPZ、虚拟环境，以及整个 `experiments/`。
- 输出目录不存在时，核心脚本会自行创建；复现不依赖已有的实验结果或样本标签。
