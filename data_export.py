from pathlib import Path
import sqlite3
import pandas as pd

root = Path(__file__).resolve().parent
db = root / "data/OpenDD/rdb2/trajectories_rdb2_v3.sqlite"
output_file = root / "data/rdb2_cars.csv"

output_file.parent.mkdir(parents=True, exist_ok=True)

# 字段mapping:
column_names = {
    "OBJID": "vehicle_id", # 车辆的唯一标识符
    "TIMESTAMP": "timestamp_s", # 时间戳（秒）
    "UTM_X": "x_m", # 车辆在地图上的X坐标
    "UTM_Y": "y_m", # 车辆在地图上的Y坐标
    "UTM_ANGLE": "heading_rad", # 车辆朝向角度
    "V": "speed_mps", # 行驶速度
    "ACC_TAN": "acc_lon_mps2", # 车辆沿行驶方向的加速度
    "ACC_LAT": "acc_lat_mps2" # 横向加速度：转弯时侧向运动变化
}

output_columns = [
    "recording", # 轨迹记录的来源
    "vehicle_id", # 车辆的唯一标识符
    "elapsed_s", # 经过时间（秒）
    "timestamp_s", # 时间戳（秒）
    "x_m", # 车辆在地图上的X坐标（米）
    "y_m", # 车辆在地图上的Y坐标（米）
    "heading_rad", # 车辆朝向角度
    "speed_mps", # 行驶速度
    "acc_lon_mps2", # 车辆沿行驶方向的加速度
    "acc_lat_mps2" # 横向加速度：转弯时侧向运动变化
]

total_trajectories = 0
total_rows = 0


with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
    # 获取全部录像表名
    table_rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    recordings =[
        row[0] for row in table_rows
        if row[0].startswith("rdb2_")
    ]
    if not recordings:
        raise ValueError("数据库中没有 rdb2 录像表")
    print(f"数据库中共有 {len(recordings)} 个 rdb2 录像表: {recordings}")
    
    # 输出文件只打开一次，重新运行覆盖导出结果
    with output_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        # 先写表头
        pd.DataFrame(columns=output_columns).to_csv(csv_file, index=False)
        # 逐段录像读取，避免同时加载全部数据
        for recording in recordings:
            # 对SQL表中引号进行转义，避免SQL语法错误
            table_name = recording.replace('"', '""')
            # 提取这段录像中的全部Car
            data = pd.read_sql_query(f"""
                SELECT OBJID, TIMESTAMP, UTM_X, UTM_Y, UTM_ANGLE, V, ACC_TAN, ACC_LAT
                FROM "{table_name}"
                WHERE CLASS = ?
                ORDER BY OBJID, TIMESTAMP
            """, conn, params=("Car",))
            if data.empty:
                print(f"录像 {recording} 中没有车辆轨迹数据")
                continue
            # 计算每辆车的首次观测时间
            first_time = (
                data.groupby("OBJID")["TIMESTAMP"]
                .transform("min")
            )
            data["elapsed_s"] = data["TIMESTAMP"] - first_time
            # 重命名字段
            data = data.rename(columns=column_names)
            data.insert(0, "recording", recording) # 轨迹记录的来源
            data = data[output_columns] # 只保留需要的字段
            # 追加写入CSV文件
            data.to_csv(csv_file, index=False, header=False)
            # 统计总轨迹数
            total_trajectories += data["vehicle_id"].nunique()
            # 统计总行数
            total_rows += len(data)
            
            print(f"已导出 {recording} 中 {data['vehicle_id'].nunique()} 辆车的轨迹，共 {len(data)} 行数据")

print(f"总计导出 {total_trajectories} 辆车的轨迹，共 {total_rows} 行数据")
