from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
map_file = root / "data/OpenDD/rdb2/map_rdb2/map_rdb2.sqlite"
trajectory_file = root / "data/rdb2_cars.csv"

# 读取地图中的车道参考线
with sqlite3.connect(map_file.as_uri() + "?mode=ro", uri=True) as conn:
    lanes = pd.read_sql_query("SELECT id, geometry FROM rdb2 WHERE type = 'trafficLane'", conn)
print(f"地图中共有 {len(lanes)} 条车道参考线")

# 读取车辆轨迹数据
data = pd.read_csv(trajectory_file, usecols=["recording", "vehicle_id", "timestamp_s", "x_m", "y_m"])

fig, ax = plt.subplots(figsize=(10, 10))

for lane in lanes.itertuples(index=False):
    # 取出括号中的坐标文本，格式为 "x1 y1, x2 y2, ..."
    coordinate_text = lane.geometry.split("(", 1)[1].rsplit(")", 1)[0]
    
    # 转化为二维坐标数组
    points = np.array([
        [float(value) for value in point.strip().split()]
        for point in coordinate_text.split(",")
    ])
    
    # 灰色线表示地图中的车道参考线
    ax.plot(points[:, 0], points[:, 1], color="lightgray", linewidth=1.5)
    
    
recording = "rdb2_154"
vehicle_id = 32881
track = data.loc[(data["recording"] == recording) & (data["vehicle_id"] == vehicle_id)].sort_values("timestamp_s")
if track.empty:
    raise ValueError(f"在录像 {recording} 中找不到车辆 {vehicle_id} 的轨迹数据")

ax.plot(track["x_m"], track["y_m"], color="royalblue", linewidth=2, label="trajectory")

# 绿色表示首次观测位置，红色表示最后观测位置
ax.scatter(
    track["x_m"].iloc[0],
    track["y_m"].iloc[0],
    color="green",
    s=70,
    zorder=5,
    label="First observation",
)

ax.scatter(
    track["x_m"].iloc[-1],
    track["y_m"].iloc[-1],
    color="red",
    s=70,
    zorder=5,
    label="Last observation",
)


# 6. 设置坐标和图例
# X、Y 使用相同比例，避免把环岛形状拉伸
ax.set_aspect("equal")
ax.set_xlabel("UTM X (m)")
ax.set_ylabel("UTM Y (m)")
ax.set_title(f"{recording} / Vehicle {vehicle_id}")
ax.ticklabel_format(style="plain", useOffset=False)
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
output_file = root / "data/rdb2_map_example.png"
fig.savefig(output_file, dpi=160)

print(f"图片已保存：{output_file}")

plt.show()
