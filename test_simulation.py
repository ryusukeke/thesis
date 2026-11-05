"""待ち時間の定義、改ざんの分離、物理的な制約を小さな手計算例で確認する。"""

import unittest

import numpy as np
import pandas as pd

from simulation import (
    POLICIES, apply_attack, assign_berths, calculate_metrics, create_normal_plan,
    create_plan, execute_adjusted_plan, execute_fixed_plan, generate_ship_data, run_experiment,
)


def small_data(arrivals, handling):
    return pd.DataFrame({
        "ship": [f"V{i + 1}" for i in range(len(arrivals))],
        "actual_arrival_time": np.array(arrivals, dtype=float),
        "reported_arrival_time": np.array(arrivals, dtype=float),
        "handling_time_B1": np.array(handling, dtype=float),
    })


class SimulationTests(unittest.TestCase):
    def test_arrivals_are_coordinated_before_attack_without_changing_input(self):
        original = small_data([0, 2, 3], [3, 3, 3])
        snapshot = original.copy(deep=True)
        data, plan = create_normal_plan(original)
        pd.testing.assert_frame_equal(original, snapshot)
        np.testing.assert_allclose(data["candidate_arrival_time"], [0, 2, 3])
        np.testing.assert_allclose(data["actual_arrival_time"], [0, 3, 6])
        np.testing.assert_allclose(data["reported_arrival_time"], [0, 3, 6])
        np.testing.assert_allclose(data["arrival_adjustment_time"], [0, 1, 3])
        pd.testing.assert_series_equal(data["handling_time_B1"], original["handling_time_B1"])
        for policy in POLICIES:
            schedule = (execute_fixed_plan(plan) if policy == "fixed"
                        else execute_adjusted_plan(data, plan, policy))
            np.testing.assert_allclose(schedule["waiting_time"], 0)
            np.testing.assert_allclose(schedule["start_time"], [0, 3, 6])

    def test_early_report_propagates_to_other_ship_from_zero_wait_baseline(self):
        data, normal = create_normal_plan(small_data([0, 2, 3], [3, 3, 3]))
        attacked = apply_attack(data, "V3", -4)
        plan = create_plan(attacked)
        fixed = execute_fixed_plan(plan).set_index("ship")
        self.assertEqual(fixed.loc["V2", "waiting_time"], 6)
        self.assertEqual(fixed.loc["V3", "actual_arrival_time"], 6)
        np.testing.assert_allclose(execute_fixed_plan(normal)["waiting_time"], 0)
        for policy in ("same_berth", "reassign"):
            adjusted = execute_adjusted_plan(attacked, plan, policy)
            np.testing.assert_allclose(adjusted["waiting_time"], 0)

    def test_late_report_keeps_coordinated_actual_arrival_and_can_be_mitigated(self):
        data, _ = create_normal_plan(small_data([0, 2, 3], [3, 3, 3]))
        attacked = apply_attack(data, "V2", 4)
        plan = create_plan(attacked)
        fixed = execute_fixed_plan(plan).set_index("ship")
        self.assertEqual(fixed.loc["V2", "reported_arrival_time"], 7)
        self.assertEqual(fixed.loc["V2", "actual_arrival_time"], 3)
        self.assertEqual(fixed.loc["V2", "waiting_time"], 6)
        for policy in ("same_berth", "reassign"):
            np.testing.assert_allclose(execute_adjusted_plan(attacked, plan, policy)["waiting_time"], 0)

    def test_reassignment_uses_handling_time_of_alternative_berth(self):
        candidates = small_data([0, 0, 1], [4, 1, 1])
        candidates["handling_time_B2"] = [10.0, 3.0, 10.0]
        data, normal = create_normal_plan(candidates)
        normal_execution = execute_adjusted_plan(data, normal, "reassign")
        np.testing.assert_allclose(normal_execution["waiting_time"], 0)
        attacked = apply_attack(data, "V2", 4)
        plan = create_plan(attacked)
        same = execute_adjusted_plan(attacked, plan, "same_berth").set_index("ship")
        reassign = execute_adjusted_plan(attacked, plan, "reassign").set_index("ship")
        self.assertEqual(same.loc["V2", "start_time"], 4)
        self.assertEqual(reassign.loc["V2", "planned_berth"], "B1")
        self.assertEqual(reassign.loc["V2", "assigned_berth"], "B2")
        self.assertEqual(reassign.loc["V2", "start_time"], 0)
        self.assertEqual(reassign.loc["V2", "handling_time"], 3)
        self.assertEqual(reassign.loc["V2", "completion_time"], 3)

    def test_congested_candidates_still_give_zero_wait_for_all_policies(self):
        for berths in (1, 2, 3, 4):
            with self.subTest(berths=berths):
                results = run_experiment(trials=2, num_ships=30, num_berths=berths,
                                         mean_interarrival_time=0.1, policies=POLICIES)
                data = results["ship_data"]
                self.assertTrue((data["arrival_adjustment_time"] > 0).any())
                normal = results["schedules"].query("scenario == 'normal'")
                np.testing.assert_allclose(normal["waiting_time"], 0)
                np.testing.assert_allclose(normal["start_time"], normal["planned_start_time"])
                self.assertTrue((normal["assigned_berth"] == normal["planned_berth"]).all())

    def test_candidates_inside_period_are_retained_after_arrival_adjustment(self):
        settings = dict(trials=1, duration_hours=1, mean_interarrival_time=0.1,
                        num_berths=1, target_ship="V2", policies=POLICIES)
        results = run_experiment(**settings)
        repeated = run_experiment(**settings)
        for name in results:
            pd.testing.assert_frame_equal(results[name], repeated[name])
        data = results["ship_data"]
        self.assertTrue((data["candidate_arrival_time"] < 1).all())
        self.assertTrue((data["actual_arrival_time"] >= 1).any())
        self.assertTrue((data["arrival_adjustment_time"] >= 0).all())
        for _, schedule in results["schedules"].groupby(["scenario", "policy"]):
            self.assertEqual(set(schedule["ship"]), set(data["ship"]))

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
        # 共通の基準時間を係数だけで補正するため、最小係数のバースが最速になる。
        np.testing.assert_array_equal(handling.argmin(axis=1), factors.argmin(axis=1))
        self.assertEqual(set(handling.argmin(axis=1)), {0, 1, 2})
        relative_handling = handling / data["base_handling_time"].to_numpy()[:, None]
        np.testing.assert_allclose(relative_handling, factors)

    def test_factor_permutation_supports_one_two_and_four_berths(self):
        for berths in (1, 2, 4):
            with self.subTest(berths=berths):
                data = generate_ship_data(20, berths, 42)
                pd.testing.assert_frame_equal(data, generate_ship_data(20, berths, 42))
                expected = np.linspace(0.9, 1.1, berths) if berths > 1 else [1.0]
                factors = data.filter(like="berth_factor_B").to_numpy()
                np.testing.assert_allclose(np.sort(factors, axis=1), np.tile(expected, (20, 1)))

    def test_all_scenarios_have_valid_paired_schedules(self):
        results = run_experiment(trials=8, policies=POLICIES)
        schedules = results["schedules"]
        inputs = results["ship_data"].set_index(["trial", "ship"])
        self.assertEqual(len(schedules), 8 * 3 * 3 * 10)
        self.assertTrue((schedules["start_time"] >= schedules["actual_arrival_time"]).all())
        self.assertTrue((schedules["waiting_time"] >= 0).all())
        np.testing.assert_allclose(schedules["turnaround_time"], schedules["waiting_time"] + schedules["handling_time"])
        for (trial, scenario, policy), schedule in schedules.groupby(["trial", "scenario", "policy"]):
            self.assertEqual(schedule["ship"].nunique(), 10)
            for _, row in schedule.iterrows():
                original = inputs.loc[(trial, row["ship"])]
                self.assertEqual(row["actual_arrival_time"], original["actual_arrival_time"])
                self.assertEqual(row["handling_time"], original[f"handling_time_{row['assigned_berth']}"])
            for _, berth in schedule.groupby("assigned_berth"):
                ordered = berth.sort_values("start_time")
                self.assertTrue((ordered["start_time"].to_numpy()[1:] >= ordered["completion_time"].to_numpy()[:-1]).all())
            if scenario == "normal":
                np.testing.assert_allclose(schedule["start_time"], schedule["planned_start_time"])
                np.testing.assert_allclose(schedule["waiting_time"], 0)
        # 条件別の計画はpolicyで複製せず、全ルールが同じ割当を参照する。
        plans = results["plans"].set_index(["trial", "scenario", "ship"])
        for _, row in schedules.iterrows():
            plan = plans.loc[(row["trial"], row["scenario"], row["ship"])]
            self.assertEqual(row["planned_berth"], plan["assigned_berth"])
            self.assertEqual(row["planned_start_time"], plan["planned_start_time"])
            original = inputs.loc[(row["trial"], row["ship"])]
            expected_report = original["actual_arrival_time"]
            if row["ship"] == "V5":
                shift = {"normal": 0, "early": -4, "late": 4}[row["scenario"]]
                expected_report = max(0, expected_report + shift)
            self.assertEqual(row["reported_arrival_time"], expected_report)

    def test_zero_shift_reproduces_normal_results_and_differences(self):
        results = run_experiment(trials=3, shift_hours=0, policies=POLICIES)
        schedules = results["schedules"]
        np.testing.assert_allclose(schedules["waiting_time_change"], 0)
        np.testing.assert_allclose(schedules["turnaround_time_change"], 0)
        comparisons = results["paired_comparison"]
        np.testing.assert_allclose(comparisons["mean_difference"], 0)
        self.assertTrue((comparisons["unchanged_trials"] == 3).all())
        for _, group in results["plans"].groupby("trial"):
            normal = group.query("scenario == 'normal'").drop(columns="scenario").reset_index(drop=True)
            for scenario in ("early", "late"):
                other = group[group["scenario"] == scenario].drop(columns="scenario").reset_index(drop=True)
                pd.testing.assert_frame_equal(normal, other)

    def test_invalid_experiment_settings_are_rejected(self):
        for settings in ({"trials": 0}, {"num_ships": 1}, {"num_berths": 0},
                         {"seed": -1}, {"target_ship": "V99"},
                         {"shift_hours": -1}, {"shift_hours": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_experiment(**settings)


if __name__ == "__main__":
    unittest.main()
