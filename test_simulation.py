"""手計算、情報の分離、物理制約、対応比較、保存済み参照値の整合を確認する。"""

import json
from contextlib import redirect_stderr
import inspect
import io
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import numpy as np
import pandas as pd

from cli import create_parser, parse_args
from result_output import REFERENCE_DIR, create_config, save_results
from simulation import (POLICIES, apply_attack, create_normal_plan, create_plan,
                        execute_adjusted_plan, execute_fixed_plan, generate_ship_data,
                        run_experiment, select_targets)


def small_data(arrivals, handling):
    return pd.DataFrame({"ship": [f"V{i+1}" for i in range(len(arrivals))],
                         "actual_arrival_time": np.array(arrivals, dtype=float),
                         "reported_arrival_time": np.array(arrivals, dtype=float),
                         "handling_time_B1": np.array(handling, dtype=float)})


class SimulationTests(unittest.TestCase):
    def test_default_trials_and_output_preserve_existing_result_files(self):
        args = create_parser().parse_args([])
        self.assertEqual(args.trials, 100)
        self.assertEqual(inspect.signature(run_experiment).parameters['trials'].default, 100)
        self.assertEqual(args.output, Path('results'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'archive').mkdir()
            arguments = ['main.py', '--output', str(output)]
            with patch('sys.argv', arguments):
                self.assertEqual(parse_args(create_parser()).output, output)
            saved = output / 'config.json'
            saved.write_text('existing result')
            with patch('sys.argv', arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(create_parser())
            self.assertEqual(saved.read_text(), 'existing result')

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
                        execute_adjusted_plan(data, plan, policy))
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

    def test_both_adjusted_policies_start_without_planned_time_limit(self):
        data, _ = create_normal_plan(small_data([0, 2, 3], [3, 3, 3]))
        attacked = apply_attack(data, 'V3', -4)
        plan = create_plan(attacked)
        for policy in ('same_berth', 'reassign'):
            with self.subTest(policy=policy):
                schedule = execute_adjusted_plan(attacked, plan, policy).set_index('ship')
                # 計画開始6時のV2を到着した3時に開始し、未到着のV3を待たない。
                self.assertEqual(schedule.loc['V2', 'start_time'], 3)
                self.assertEqual(schedule.loc['V3', 'start_time'], 6)
                np.testing.assert_array_equal(schedule.waiting_time, [0, 0, 0])
                # 遅い申告でも2時間を超える前倒しを許し、到着時点で開始する。
                late = apply_attack(data, 'V2', 8)
                late_schedule = execute_adjusted_plan(late, create_plan(late), policy).set_index('ship')
                self.assertEqual(late_schedule.loc['V2', 'planned_start_time'], 11)
                self.assertEqual(late_schedule.loc['V2', 'start_time'], 3)

    def test_reassignment_uses_free_berth_and_its_handling_time(self):
        raw = small_data([0, 0, 1], [4, 1, 1])
        raw['handling_time_B2'] = [10., 3., 10.]
        data, _ = create_normal_plan(raw)
        attacked = apply_attack(data, 'V2', 4)
        plan = create_plan(attacked)
        same = execute_adjusted_plan(attacked, plan, 'same_berth').set_index('ship')
        changed = execute_adjusted_plan(attacked, plan, 'reassign').set_index('ship')
        self.assertEqual(same.loc['V2', 'start_time'], 4)
        self.assertEqual(changed.loc['V2', 'planned_berth'], 'B1')
        self.assertEqual(changed.loc['V2', 'assigned_berth'], 'B2')
        self.assertEqual(changed.loc['V2', 'start_time'], 0)
        self.assertEqual(changed.loc['V2', 'handling_time'], 3)
        self.assertEqual(changed.loc['V2', 'completion_time'], 3)

    def test_early_start_does_not_interrupt_handling_for_later_ship(self):
        data = small_data([0, 4], [10, 2])
        attacked = apply_attack(data, 'V1', 8)
        plan = create_plan(attacked)
        fixed = execute_fixed_plan(plan).set_index('ship')
        self.assertEqual(fixed.loc['V2', 'waiting_time'], 0)
        for policy in ('same_berth', 'reassign'):
            schedule = execute_adjusted_plan(attacked, plan, policy).set_index('ship')
            self.assertEqual(schedule.loc['V1', 'start_time'], 0)
            self.assertEqual(schedule.loc['V1', 'completion_time'], 10)
            self.assertEqual(schedule.loc['V2', 'start_time'], 10)
            self.assertEqual(schedule.loc['V2', 'waiting_time'], 6)

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

    def test_full_study_counts_targets_and_three_policies_share_inputs(self):
        result = run_experiment(trials=1, full_study=True)
        runs = result['run_metrics']
        self.assertEqual(len(result['plan_diagnostics']), 3)
        self.assertEqual(POLICIES, ('fixed', 'same_berth', 'reassign'))
        self.assertEqual(len(runs.query("scenario!='normal'")), 162)
        self.assertEqual(set(runs.experiment_group), {'main'})
        self.assertNotIn('early_start_limit_hours', runs.columns)
        reassign = result['schedules'].query("policy=='reassign'")
        np.testing.assert_allclose(reassign.waiting_time, 0)
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
                         {'shift_hours':-1}, {'shift_hours':float('nan')},
                         {'duration_hours':0}, {'mean_interarrival_time':0}, {'policies':('bad',)},
                         {'policies':('unrestricted',)},
                         {'duration_hours':None,'num_ships':1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_experiment(**settings)

    def test_saved_kobe_reference_matches_daily_counts_and_selected_rows(self):
        reference = json.loads((REFERENCE_DIR / 'kobe_reference.json').read_text())
        daily = pd.read_csv(REFERENCE_DIR / 'kobe_daily_forecasts.csv')
        selected = pd.read_csv(REFERENCE_DIR / 'kobe_reference_rows.csv')
        counted = selected[selected['counted']]
        self.assertEqual(reference['observed_snapshot_days'], len(daily))
        self.assertEqual(reference['scheduled_ship_days'], int(daily.scheduled_ship_days.sum()))
        self.assertEqual(reference['scheduled_ship_days'], len(counted))
        identity = counted.call_sign.fillna('').str.strip()
        identity = identity.mask(identity.eq(''), counted.ship_name.fillna('').str.strip())
        self.assertFalse(counted.assign(identity=identity).duplicated(['snapshot_date', 'identity', 'forecast_date']).any())
        for row in daily.itertuples():
            self.assertEqual(row.scheduled_ship_days, int((counted.snapshot_date == row.snapshot_date).sum()))
        self.assertEqual(reference['mean_candidate_interval_hours'], 24 * len(daily) / len(counted))

    def test_saved_output_contains_only_conditions_run_metrics_and_summary(self):
        args = create_parser().parse_args(['--trials', '1', '--no-plot'])
        results = run_experiment(trials=1)
        config = create_config(args, results, 3.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_results(results, config, output)
            self.assertEqual({path.name for path in output.iterdir()},
                             {'config.json', 'run_metrics.csv', 'summary.csv'})
            for name in ('run_metrics', 'summary'):
                saved = pd.read_csv(output / f'{name}.csv')
                pd.testing.assert_frame_equal(saved, results[name], check_dtype=False)
            saved_config = json.loads((output / 'config.json').read_text())
            self.assertEqual(saved_config['first_seed'], 42)
            self.assertEqual(saved_config['policies'], ['fixed', 'same_berth', 'reassign'])
            self.assertIn('normal_plan_diagnostics', saved_config)


if __name__ == '__main__':
    unittest.main()
