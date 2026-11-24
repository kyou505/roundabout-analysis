# 数据准备

本目录不随 Git 提交原始数据库或处理后的轨迹。克隆代码后，从 [OpenDD 官方入口](https://l3pilot.eu/data/opendd.html) 获取 v3 数据并解压。

## 当前核心流程需要 rdb2

在项目根目录下放置这两个文件，保留文件名及目录层级：

```text
data/OpenDD/rdb2/trajectories_rdb2_v3.sqlite
data/OpenDD/rdb2/map_rdb2/map_rdb2.sqlite
```

先运行 `data_export.py`，再运行 `data_process.py` 和 `clustering.py`。导出 CSV、处理后的数组和距离矩阵都由代码生成，不需要从别人机器复制。

数据库版本通过同目录下的 `SHA256SUMS` 核对；其中路径相对于项目根目录。macOS 可执行：

```bash
shasum -a 256 data/OpenDD/rdb2/trajectories_rdb2_v3.sqlite
shasum -a 256 data/OpenDD/rdb2/map_rdb2/map_rdb2.sqlite
```

将两行结果与 `data/SHA256SUMS` 对比。Linux 可使用 `sha256sum`。仅下载 rdb2 时，无需准备清单中的其他路口。

## 其他本地数据

校验清单也记录了研究中使用的 rdb3、rdb4 原始文件，便于辨识数据版本。它们的探索脚本仍在本地 `experiments/` 中，当前提交的核心入口不支持自动复现这些探索实验。

```text
data/rdb3/trajectories_rdb3_v3.sqlite
data/rdb3/map_rdb3/map_rdb3.sqlite
data/rdb4/trajectories_rdb4_v3.sqlite
data/rdb4/map_rdb4/map_rdb4.sqlite
```

## 数据来源和许可

引用：Antonia Breuer, Jan-Aike Termöhlen, Silviu Homoceanu, Tim Fingscheidt. *OpenDD: A Large-Scale Roundabout Drone Dataset*. IEEE ITSC, 2020。引用格式以官方入口为准。

OpenDD 官方标明数据许可为 [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)，下载包附有 `LICENSE_by_nd_4.0.txt`。下载后保留数据包原附的许可和引用说明。该数据许可与本项目代码的许可是不同的。

本仓库提供处理代码和数据来源，不重新发布原始或改造后的数据包。原始轨迹 SQLite 单文件约 352–481 MiB，超过 GitHub 普通 Git 的 [100 MiB 单文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
