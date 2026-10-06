"""待ち時間の定義、改ざんの分離、物理的な制約を小さな手計算例で確認する。"""

import unittest

import numpy as np
import pandas as pd

from simulation import apply_attack, assign_berths, calculate_metrics, generate_ship_data, run_experiment


def small_data(arrivals, handling):
    return pd.DataFrame({
        "ship": [f"V{i + 1}" for i in range(len(arrivals))],
        "actual_arrival_time": np.array(arrivals, dtype=float),
        "reported_arrival_time": np.array(arrivals, dtype=float),
        "handling_time_B1": np.array(handling, dtype=float),
    })


class SimulationTests(unittest.TestCase):
    def test_normal_fcfs_matches_hand_calculation(self):
        schedule = assign_berths(small_data([0, 2, 3], [3, 3, 3]))
        np.testing.assert_allclose(schedule["start_time"], [0, 3, 6])
        np.testing.assert_allclose(schedule["waiting_time"], [0, 1, 3])
        np.testing.assert_allclose(schedule["turnaround_time"], [3, 4, 6])
        metrics = calculate_metrics(schedule, "V2")
        self.assertEqual(metrics["total_waiting_time"], 4)
        self.assertEqual(metrics["total_turnaround_time"], 13)
        self.assertEqual(metrics["target_waiting_time"], 1)

    def test_early_report_changes_order_but_cannot_start_before_arrival(self):
        data = small_data([0, 5, 2], [1, 2, 1])
        schedule = assign_berths(apply_attack(data, "V2", -4))
        self.assertEqual(schedule["ship"].tolist(), ["V1", "V2", "V3"])
        np.testing.assert_allclose(schedule["planned_start_time"], [0, 1, 3])
        np.testing.assert_allclose(schedule["start_time"], [0, 5, 7])
        np.testing.assert_allclose(schedule["waiting_time"], [0, 0, 5])

    def test_late_report_uses_actual_arrival_to_measure_wait(self):
        data = small_data([0, 2, 3], [3, 3, 3])
        schedule = assign_berths(apply_attack(data, "V2", 4)).set_index("ship")
        self.assertEqual(schedule.loc["V2", "reported_arrival_time"], 6)
        self.assertEqual(schedule.loc["V2", "start_time"], 6)
        self.assertEqual(schedule.loc["V2", "waiting_time"], 4)
        self.assertEqual(schedule.loc["V3", "waiting_time"], 0)

    def test_attack_preserves_input_and_only_changes_target_report(self):
        data = generate_ship_data()
        original = data.copy(deep=True)
        attacked = apply_attack(data, "V5", -4)
        pd.testing.assert_frame_equal(data, original)
        columns = [c for c in data if c != "reported_arrival_time"]
        pd.testing.assert_frame_equal(data[columns], attacked[columns])
        other = data["ship"] != "V5"
        pd.testing.assert_series_equal(data.loc[other, "reported_arrival_time"], attacked.loc[other, "reported_arrival_time"])
        clamped = apply_attack(data, "V1", -4)
        self.assertEqual(clamped.iloc[0]["reported_arrival_time"], 0)

    def test_planner_does_not_use_true_arrival_to_select_berth(self):
        data = small_data([0, 2, 3], [3, 3, 3])
        changed = data.copy()
        changed.loc[1, "actual_arrival_time"] = 20
        columns = ["ship", "assigned_berth", "planned_start_time", "planned_completion_time"]
        pd.testing.assert_frame_equal(assign_berths(data)[columns], assign_berths(changed)[columns])
        self.assertEqual(assign_berths(changed).iloc[1]["start_time"], 20)

    def test_berth_selection_and_tie_break_are_deterministic(self):
        data = small_data([0], [3])
        data["handling_time_B2"] = 2.0
        self.assertEqual(assign_berths(data).iloc[0]["assigned_berth"], "B2")
        data["handling_time_B2"] = 3.0
        self.assertEqual(assign_berths(data).iloc[0]["assigned_berth"], "B1")

    def test_generated_data_are_reproducible_and_correlated(self):
        data = generate_ship_data(1000, 3, 42)
        pd.testing.assert_frame_equal(data, generate_ship_data(1000, 3, 42))
        handling = data.filter(like="handling_time_B")
        self.assertTrue((handling.to_numpy() > 0).all())
        self.assertGreater(handling.corr().loc["handling_time_B1", "handling_time_B3"], 0.95)

    def test_each_ship_has_permuted_factors_and_different_fastest_berths(self):
        data = generate_ship_data(100, 3, 42)
        factors = data.filter(like="berth_factor_B").to_numpy()
        np.testing.assert_allclose(np.sort(factors, axis=1), np.tile([0.9, 1.0, 1.1], (100, 1)))
        handling = data.filter(like="handling_time_B").to_numpy()
        # 係数0.9の誤差上限は0.945で、係数1.0の下限0.95より小さい。
        np.testing.assert_array_equal(handling.argmin(axis=1), factors.argmin(axis=1))
        self.assertEqual(set(handling.argmin(axis=1)), {0, 1, 2})
        relative_noise = handling / (data["base_handling_time"].to_numpy()[:, None] * factors) - 1
        self.assertTrue((np.abs(relative_noise) <= 0.05).all())

    def test_factor_permutation_supports_one_two_and_four_berths(self):
        for berths in (1, 2, 4):
            with self.subTest(berths=berths):
                data = generate_ship_data(20, berths, 42)
                pd.testing.assert_frame_equal(data, generate_ship_data(20, berths, 42))
                expected = np.linspace(0.9, 1.1, berths) if berths > 1 else [1.0]
                factors = data.filter(like="berth_factor_B").to_numpy()
                np.testing.assert_allclose(np.sort(factors, axis=1), np.tile(expected, (20, 1)))

    def test_all_scenarios_have_valid_paired_schedules(self):
        results = run_experiment(trials=8)
        schedules = results["schedules"]
        inputs = results["ship_data"].set_index(["trial", "ship"])
        self.assertEqual(len(schedules), 8 * 3 * 10)
        self.assertTrue((schedules["start_time"] >= schedules["actual_arrival_time"]).all())
        self.assertTrue((schedules["waiting_time"] >= 0).all())
        np.testing.assert_allclose(schedules["turnaround_time"], schedules["waiting_time"] + schedules["handling_time"])
        for (trial, scenario), schedule in schedules.groupby(["trial", "scenario"]):
            self.assertEqual(schedule["ship"].nunique(), 10)
            for _, row in schedule.iterrows():
                original = inputs.loc[(trial, row["ship"])]
                self.assertEqual(row["actual_arrival_time"], original["actual_arrival_time"])
                self.assertEqual(row["handling_time"], original[f"handling_time_{row['assigned_berth']}"])
            for _, berth in schedule.groupby("assigned_berth"):
                ordered = berth.sort_values("assignment_order")
                self.assertTrue((ordered["start_time"].to_numpy()[1:] >= ordered["completion_time"].to_numpy()[:-1]).all())
            if scenario == "normal":
                np.testing.assert_allclose(schedule["start_time"], schedule["planned_start_time"])

    def test_zero_shift_reproduces_normal_results_and_differences(self):
        results = run_experiment(trials=3, shift_hours=0)
        schedules = results["schedules"]
        np.testing.assert_allclose(schedules["waiting_time_change"], 0)
        np.testing.assert_allclose(schedules["turnaround_time_change"], 0)
        comparisons = results["paired_comparison"]
        np.testing.assert_allclose(comparisons["mean_difference"], 0)
        self.assertTrue((comparisons["unchanged_trials"] == 3).all())

    def test_invalid_experiment_settings_are_rejected(self):
        for settings in ({"trials": 0}, {"num_ships": 1}, {"num_berths": 0},
                         {"seed": -1}, {"target_ship": "V99"},
                         {"shift_hours": -1}, {"shift_hours": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_experiment(**settings)


if __name__ == "__main__":
    unittest.main()
