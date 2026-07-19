import numpy as np
import pandas as pd

np.random.seed(42)

num_ships = 10
num_berths = 4

# 到着間隔の平均値
mean_interarrival_time = 2.37

# 荷役時間の分布パラメータ
handling_shape = 2.0
handling_scale = 2.4

# 到着間隔を指数分布から生成
interarrival_times = np.random.exponential(
    scale=mean_interarrival_time,
    size=num_ships
)

# 最初の船を時刻0にする
arrival_times = np.cumsum(interarrival_times)
arrival_times = arrival_times - arrival_times[0]

# 船とバースの組み合わせごとの荷役時間
handling_times = np.random.gamma(
    shape=handling_shape,
    scale=handling_scale,
    size=(num_ships, num_berths)
)

arrival_times = np.round(arrival_times, 1)
handling_times = np.round(handling_times, 1)

# 入力データを作成
data = {
    "ship": [f"V{i + 1}" for i in range(num_ships)],
    "arrival_time_A": arrival_times,
}

for berth_index in range(num_berths):
    data[f"handling_time_B{berth_index + 1}"] = (
        handling_times[:, berth_index]
    )

df = pd.DataFrame(data)

print("生成された船舶データ")
print(df.to_string(index=False))

# ==========================================
# FCFSによるバース割当
# ==========================================

# 各バースが利用可能になる時刻
berth_available_times = np.zeros(num_berths)

schedule = []

# 到着時刻順に並べる
fcfs_df = df.sort_values(
    by=["arrival_time_A", "ship"]
).reset_index(drop=True)

for _, row in fcfs_df.iterrows():
    ship = row["ship"]
    arrival_time = row["arrival_time_A"]

    berth_candidates = []

    for berth_index in range(num_berths):
        berth_name = f"B{berth_index + 1}"

        handling_time = row[
            f"handling_time_{berth_name}"
        ]

        # 荷役開始可能時刻
        start_time = max(
            arrival_time,
            berth_available_times[berth_index]
        )

        completion_time = start_time + handling_time

        berth_candidates.append({
            "berth_index": berth_index,
            "berth": berth_name,
            "handling_time": handling_time,
            "start_time": start_time,
            "completion_time": completion_time
        })

    # 開始時刻 → 完了時刻 → バース番号の順で選択
    selected = min(
        berth_candidates,
        key=lambda candidate: (
            candidate["start_time"],
            candidate["completion_time"],
            candidate["berth_index"]
        )
    )

    selected_berth_index = selected["berth_index"]

    berth_available_times[selected_berth_index] = (
        selected["completion_time"]
    )

    waiting_time = selected["start_time"] - arrival_time

    schedule.append({
        "ship": ship,
        "arrival_time": arrival_time,
        "assigned_berth": selected["berth"],
        "handling_time": selected["handling_time"],
        "start_time": selected["start_time"],
        "completion_time": selected["completion_time"],
        "waiting_time": waiting_time
    })

schedule_df = pd.DataFrame(schedule)

time_columns = [
    "arrival_time",
    "handling_time",
    "start_time",
    "completion_time",
    "waiting_time"
]

schedule_df[time_columns] = schedule_df[time_columns].round(1)

print("\nFCFSによるバース割当結果")
print(schedule_df.to_string(index=False))

# ==========================================
# 評価指標
# ==========================================

total_waiting_time = schedule_df["waiting_time"].sum()
average_waiting_time = schedule_df["waiting_time"].mean()

schedule_df["turnaround_time"] = (
    schedule_df["completion_time"]
    - schedule_df["arrival_time"]
)

total_turnaround_time = schedule_df["turnaround_time"].sum()
average_turnaround_time = schedule_df["turnaround_time"].mean()

print("\n評価指標")
print(f"総待ち時間: {total_waiting_time:.1f}時間")
print(f"平均待ち時間: {average_waiting_time:.1f}時間")
print(f"総在港時間: {total_turnaround_time:.1f}時間")
print(f"平均在港時間: {average_turnaround_time:.1f}時間")