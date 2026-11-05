"""手計算、情報の分離、物理制約、対応比較、神戸港予定の集計を確認する。"""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from port_collection.analyze_port_data import analyze_forecasts
from simulation import (POLICIES, apply_attack, create_normal_plan, create_plan,
                        execute_adjusted_plan, execute_fixed_plan, generate_ship_data,
                        run_experiment, select_targets)


def small_data(arrivals, handling):
    return pd.DataFrame({"ship": [f"V{i+1}" for i in range(len(arrivals))],
                         "actual_arrival_time": np.array(arrivals, dtype=float),
                         "reported_arrival_time": np.array(arrivals, dtype=float),
                         "handling_time_B1": np.array(handling, dtype=float)})


class SimulationTests(unittest.TestCase):
    def test_coordinated_normal_plan_matches_hand_calculation_and_preserves_input(self):
        raw = small_data([0, 2, 3], [3, 3, 3])
        saved = raw.copy(deep=True)
        data, plan = create_normal_plan(raw)
        pd.testing.assert_frame_equal(raw, saved)
        np.testing.assert_array_equal(data.actual_arrival_time, [0, 3, 6])
        np.testing.assert_array_equal(data.candidate_arrival_time, [0, 2, 3])
        np.testing.assert_array_equal(data.arrival_adjustment_time, [0, 1, 3])
        for policy in POLICIES:
            schedule = (execute_fixed_plan(plan) if policy == "fixed" else
                        execute_adjusted_plan(data, plan, policy, None if policy == "unrestricted" else 2))
            np.testing.assert_array_equal(schedule.waiting_time, [0, 0, 0])
            np.testing.assert_array_equal(schedule.start_time, [0, 3, 6])

    def test_distance_factors_and_random_attributes_are_reproducible(self):
        for berths in (1, 4):
            data = generate_ship_data(20, berths, 42)
            pd.testing.assert_frame_equal(data, generate_ship_data(20, berths, 42))
            for b in range(1, berths + 1):
                expected = 1 + 0.1 * (data.cargo_nearest_berth - b).abs()
                np.testing.assert_allclose(data[f'berth_factor_B{b}'], expected)
                np.testing.assert_allclose(data[f'handling_time_B{b}'], data.base_handling_time * expected)
        sparse = generate_ship_data(num_berths=4, seed=42, duration_hours=168, mean_interarrival_time=4)
        dense = generate_ship_data(num_berths=4, seed=42, duration_hours=168, mean_interarrival_time=2)
        cols = ['base_handling_time', 'cargo_nearest_berth']
        pd.testing.assert_frame_equal(sparse[cols], dense.iloc[:len(sparse)][cols])
        self.assertTrue((dense.base_handling_time > 0).all())

    def test_attack_changes_only_one_report_and_clamps_at_zero(self):
        data, _ = create_normal_plan(generate_ship_data())
        saved = data.copy(deep=True)
        attacked = apply_attack(data, 'V5', -4)
        pd.testing.assert_frame_equal(data, saved)
        cols = [c for c in data if c != 'reported_arrival_time']
        pd.testing.assert_frame_equal(data[cols], attacked[cols])
        pd.testing.assert_series_equal(data.loc[data.ship != 'V5', 'reported_arrival_time'],
                                       attacked.loc[attacked.ship != 'V5', 'reported_arrival_time'])
        self.assertEqual(apply_attack(data, 'V1', -4).iloc[0].reported_arrival_time, 0)

    def test_planner_does_not_use_future_actual_arrival(self):
        data = small_data([0, 2, 3], [3, 3, 3])
        changed = data.copy()
        changed.loc[1, 'actual_arrival_time'] = 20
        cols = ['ship', 'assigned_berth', 'planned_start_time', 'planned_completion_time']
        pd.testing.assert_frame_equal(create_plan(data)[cols], create_plan(changed)[cols])
        self.assertEqual(execute_fixed_plan(create_plan(changed)).set_index('ship').loc['V2', 'start_time'], 20)

    def test_fixed_early_and_late_reports_match_hand_calculation(self):
        data, _ = create_normal_plan(small_data([0, 2, 3], [3, 3, 3]))
        early = execute_fixed_plan(create_plan(apply_attack(data, 'V3', -4))).set_index('ship')
        self.assertEqual(early.loc['V3', 'start_time'], 6)
        self.assertEqual(early.loc['V2', 'waiting_time'], 6)
        late = execute_fixed_plan(create_plan(apply_attack(data, 'V2', 4))).set_index('ship')
        self.assertEqual(late.loc['V2', 'actual_arrival_time'], 3)
        self.assertEqual(late.loc['V2', 'reported_arrival_time'], 7)
        self.assertEqual(late.loc['V2', 'waiting_time'], 6)

    def test_advance_release_is_an_event_even_without_arrival_or_completion(self):
        data, _ = create_normal_plan(small_data([0, 2, 3], [3, 3, 3]))
        attacked = apply_attack(data, 'V3', -4)
        plan = create_plan(attacked)
        bounded = execute_adjusted_plan(attacked, plan, 'same_berth', 2).set_index('ship')
        # 4時には到着も完了もないが、V2の開始制限が解除される。
        self.assertEqual(bounded.loc['V2', 'start_time'], 4)
        self.assertEqual(bounded.loc['V2', 'waiting_time'], 1)
        self.assertEqual(bounded.loc['V3', 'start_time'], 7)
        free = execute_adjusted_plan(attacked, plan, 'unrestricted', None)
        np.testing.assert_array_equal(free.waiting_time, [0, 0, 0])
        zero = execute_adjusted_plan(attacked, plan, 'reassign', 0)
        self.assertTrue((zero.start_time >= zero.planned_start_time).all())

    def test_reassignment_uses_selected_berth_time_and_respects_advance_limit(self):
        raw = small_data([0, 0, 1], [4, 1, 1])
        raw['handling_time_B2'] = [10., 3., 10.]
        data, _ = create_normal_plan(raw)
        attacked = apply_attack(data, 'V2', 4)
        plan = create_plan(attacked)
        same = execute_adjusted_plan(attacked, plan, 'same_berth', 2).set_index('ship')
        changed = execute_adjusted_plan(attacked, plan, 'reassign', 2).set_index('ship')
        self.assertEqual(same.loc['V2', 'start_time'], 4)
        self.assertEqual(changed.loc['V2', 'planned_berth'], 'B1')
        self.assertEqual(changed.loc['V2', 'assigned_berth'], 'B2')
        self.assertEqual(changed.loc['V2', 'start_time'], 2)
        self.assertEqual(changed.loc['V2', 'handling_time'], 3)
        self.assertEqual(changed.loc['V2', 'completion_time'], 5)

    def test_all_cases_preserve_physics_inputs_and_paired_metrics(self):
        results = run_experiment(trials=2, duration_hours=None, num_ships=25, mean_interarrival_time=0.3)
        data = results['ship_data'].set_index(['demand_multiplier', 'trial', 'ship'])
        schedules = results['schedules']
        keys = ['experiment_group', 'demand_multiplier', 'trial', 'target_position', 'shift_hours', 'scenario', 'policy']
        for _, case in schedules.groupby(keys):
            self.assertEqual(case.ship.nunique(), 25)
            self.assertTrue((case.start_time >= case.actual_arrival_time).all())
            self.assertTrue((case.start_time >= case.earliest_start_time).all())
            np.testing.assert_allclose(case.turnaround_time, case.waiting_time + case.handling_time)
            for berth, rows in case.groupby('assigned_berth'):
                rows = rows.sort_values('start_time')
                self.assertTrue((rows.start_time.to_numpy()[1:] >= rows.completion_time.to_numpy()[:-1]).all())
                expected = [data.loc[(r.demand_multiplier, r.trial, r.ship), f'handling_time_{berth}'] for r in rows.itertuples()]
                np.testing.assert_allclose(rows.handling_time, expected)
            for row in case.itertuples():
                self.assertEqual(row.actual_arrival_time, data.loc[(row.demand_multiplier, row.trial, row.ship), 'actual_arrival_time'])
        normal = schedules[schedules.scenario == 'normal']
        np.testing.assert_allclose(normal.waiting_time, 0)
        self.assertTrue((normal.assigned_berth == normal.planned_berth).all())
        compare = results['paired_comparison']
        for row in compare.itertuples():
            runs = results['run_metrics']
            selected = runs[(runs.experiment_group == row.experiment_group) & (runs.demand_multiplier == row.demand_multiplier)
                            & (runs.target_position == row.target_position) & (runs.shift_hours == row.shift_hours)
                            & (runs.scenario == row.scenario) & (runs.policy == row.policy)]
            self.assertAlmostEqual(row.attack_mean, selected[row.metric].mean())

    def test_zero_shift_and_repeated_seed_reproduce_results(self):
        settings = dict(trials=1, duration_hours=None, num_ships=10, shift_hours=0)
        result = run_experiment(**settings)
        repeated = run_experiment(**settings)
        for name in result:
            pd.testing.assert_frame_equal(result[name], repeated[name])
        np.testing.assert_allclose(result['paired_comparison'].mean_difference, 0)
        self.assertTrue((result['paired_comparison'].unchanged_trials == 1).all())

    def test_full_study_counts_targets_and_sensitivity_share_inputs(self):
        result = run_experiment(trials=1, full_study=True)
        runs = result['run_metrics']
        self.assertEqual(len(result['plan_diagnostics']), 3)
        self.assertEqual(len(runs.query("experiment_group=='main' and scenario!='normal'")), 216)
        self.assertEqual(len(runs.query("experiment_group=='sensitivity' and scenario!='normal'")), 6)
        self.assertEqual(set(runs.query("experiment_group=='sensitivity'").early_start_limit_hours), {0.0})
        for _, data in result['ship_data'].groupby('demand_multiplier'):
            chosen = select_targets(data, True, 'V5')
            ordered = data.sort_values('actual_arrival_time', kind='stable').ship.tolist()
            self.assertEqual([t for _, t in chosen], [ordered[int(np.ceil(len(data)*p))-1] for p in (.1,.5,.9)])
        schedules = result['schedules']
        for row in schedules.query("scenario!='normal'").itertuples():
            expected = max(0, row.actual_arrival_time + (-row.shift_hours if row.scenario=='early' else row.shift_hours)) if row.ship==row.target_ship else row.actual_arrival_time
            self.assertEqual(row.reported_arrival_time, expected)

    def test_adjusted_arrivals_after_period_are_not_dropped(self):
        result = run_experiment(trials=1, duration_hours=1, mean_interarrival_time=0.1, num_berths=1, target_ship='V2')
        data = result['ship_data']
        self.assertTrue((data.candidate_arrival_time < 1).all())
        self.assertTrue((data.actual_arrival_time >= 1).any())
        self.assertEqual(result['plan_diagnostics'].arrivals_after_period.iloc[0], int((data.actual_arrival_time >= 1).sum()))
        for _, schedule in result['schedules'].groupby(['scenario', 'policy']):
            self.assertEqual(set(schedule.ship), set(data.ship))

    def test_invalid_conditions_are_rejected_without_reselecting_trials(self):
        for settings in ({'trials':0}, {'num_berths':0}, {'seed':-1}, {'target_ship':'V9999'},
                         {'shift_hours':-1}, {'early_start_limit':-1}, {'early_start_limit':float('nan')},
                         {'duration_hours':0}, {'mean_interarrival_time':0}, {'policies':('bad',)},
                         {'duration_hours':None,'num_ships':1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_experiment(**settings)

    def test_kobe_daily_last_snapshot_next_day_counts_are_not_history_row_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'2026-10-23'
            folder.mkdir()
            header = ['コールサイン','船名','船席','入港日時']
            for stamp, rows in [('2026102307000000', [['A','ONE','PC-14','10/24 08:00']]),
                                ('2026102310000000', [['A','ONE','PC-14','10/24 09:00'],
                                                       ['A','ONE','PC-16','10/24 10:00'],
                                                       ['B','TWO','PC-17','10/24 12:00'],
                                                       ['C','OTHER','RC-7','10/24 13:00'],
                                                       ['D','LATER','PC-14','10/25 08:00']])]:
                with (folder/f'{stamp}_入港船予定表.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
            reference, daily, selected=analyze_forecasts(tmp,'2026-10-23')
            self.assertEqual(reference['scheduled_ship_days'],2)
            self.assertEqual(reference['mean_candidate_interval_hours'],12)
            self.assertEqual(daily[0]['merged_same_ship_day_rows'],1)
            self.assertEqual(len(selected),3)
            self.assertEqual(sum(row['counted'] for row in selected),2)


if __name__ == '__main__':
    unittest.main()
