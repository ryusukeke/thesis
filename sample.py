import numpy as np
import pandas as pd

np.random.seed(42)

num_ships = 10      # 船舶数
num_berths = 2      # バース数

# 到着間隔の平均値
# 神戸港の外航フルコンテナ船の年間入港隻数の8760時間 ÷ 3689隻 ≒ 2.37時間より、
mean_interarrival_time = 2.37

# 荷役時間の分布パラメータ
handling_shape = 2.0
handling_scale = 2.4    # 平均荷役時間 = 4.8

# 到着時刻 A_j の生成
# 船舶の到着間隔を指数分布から生成する
interarrival_times = np.random.exponential(
    scale=mean_interarrival_time,
    size=num_ships
)

# 最初の船は時刻0に到着するとする
arrival_times = np.cumsum(interarrival_times)
arrival_times = arrival_times - arrival_times[0]

# 荷役時間 H_jb の生成
# 各船舶について、バースごとに異なる荷役時間を生成する
# 行：船舶、列：バース
handling_times = np.random.gamma(
    shape=handling_shape,
    scale=handling_scale,
    size=(num_ships, num_berths)
)

# 小数第1位に丸める
arrival_times = np.round(arrival_times, 1)
handling_times = np.round(handling_times, 1)

# 実験用データの作成
data = {
    "ship": [f"V{i + 1}" for i in range(num_ships)],
    "arrival_time_A": arrival_times,
}

# バースごとの荷役時間を列として追加
for berth_index in range(num_berths):
    data[f"handling_time_B{berth_index + 1}"] = handling_times[:, berth_index]

df = pd.DataFrame(data)

print(df.to_string(index=False))