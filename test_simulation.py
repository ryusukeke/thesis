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
from result_output import (EXCEL_COLUMNS, EXCEL_VALUES, REFERENCE_DIR, SUMMARY_OVERVIEW_COLUMNS,
                           create_config, create_summary_overview, excel_number_format, save_results,
                           translate_excel_table)
from simulation import (POLICIES, apply_attack, create_plan,
                        calculate_metrics, execute_adjusted_plan, execute_fixed_plan, generate_ship_data,
                        run_experiment, select_targets)


def small_data(arrivals, handling):
    return pd.DataFrame({"ship": [f"V{i+1}" for i in range(len(arrivals))],
                         "actual_arrival_time": np.array(arrivals, dtype=float),
                         "reported_arrival_time": np.array(arrivals, dtype=float),
                         "handling_time_B1": np.array(handling, dtype=float)})


class SimulationTests(unittest.TestCase):
    def test_default_trials_and_output_preserve_existing_result_files(self):
        args = create_parser().parse_args([])
        self.assertEqual(args.trials, 1)
        self.assertEqual(inspect.signature(run_experiment).parameters['trials'].default, 1)
        self.assertEqual(args.output, Path('results'))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'subdirectory').mkdir()
            arguments = ['main.py', '--output', str(output)]
            with patch('sys.argv', arguments):
                self.assertEqual(parse_args(create_parser()).output, output)
            saved = output / 'config.json'
            saved.write_text('existing result')
            with patch('sys.argv', arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(create_parser())
            self.assertEqual(saved.read_text(), 'existing result')

    def test_normal_plan_preserves_arrivals_and_allows_waiting(self):
        data = small_data([0, 2, 3], [3, 3, 3])
        saved = data.copy(deep=True)
        plan = create_plan(data)
        pd.testing.assert_frame_equal(data, saved)
        np.testing.assert_array_equal(plan.actual_arrival_time, [0, 2, 3])
        for policy in POLICIES:
            schedule = (execute_fixed_plan(plan) if policy == "fixed" else
                        execute_adjusted_plan(data, plan, policy))
            np.testing.assert_array_equal(schedule.waiting_time, [0, 1, 3])
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
        data = generate_ship_data()
        saved = data.copy(deep=True)
        attacked = apply_attack(data, {'V5': -4})
        pd.testing.assert_frame_equal(data, saved)
        cols = [c for c in data if c != 'reported_arrival_time']
        pd.testing.assert_frame_equal(data[cols], attacked[cols])
        pd.testing.assert_series_equal(data.loc[data.ship != 'V5', 'reported_arrival_time'],
                                       attacked.loc[attacked.ship != 'V5', 'reported_arrival_time'])
        self.assertEqual(apply_attack(data, {'V1': -4}).iloc[0].reported_arrival_time, 0)

    def test_planner_does_not_use_future_actual_arrival(self):
        data = small_data([0, 2, 3], [3, 3, 3])
        changed = data.copy()
        changed.loc[1, 'actual_arrival_time'] = 20
        cols = ['ship', 'assigned_berth', 'planned_start_time', 'planned_completion_time']
        pd.testing.assert_frame_equal(create_plan(data)[cols], create_plan(changed)[cols])
        self.assertEqual(execute_fixed_plan(create_plan(changed)).set_index('ship').loc['V2', 'start_time'], 20)

    def test_fixed_early_and_late_reports_match_hand_calculation(self):
        data = small_data([0, 3, 6], [3, 3, 3])
        early = execute_fixed_plan(create_plan(apply_attack(data, {'V3': -4}))).set_index('ship')
        self.assertEqual(early.loc['V3', 'start_time'], 6)
        self.assertEqual(early.loc['V2', 'waiting_time'], 6)
        late = execute_fixed_plan(create_plan(apply_attack(data, {'V2': 4}))).set_index('ship')
        self.assertEqual(late.loc['V2', 'actual_arrival_time'], 3)
        self.assertEqual(late.loc['V2', 'reported_arrival_time'], 7)
        self.assertEqual(late.loc['V2', 'waiting_time'], 6)

    def test_both_adjusted_policies_start_without_planned_time_limit(self):
        data = small_data([0, 3, 6], [3, 3, 3])
        attacked = apply_attack(data, {'V3': -4})
        plan = create_plan(attacked)
        for policy in ('same_berth', 'reassign'):
            with self.subTest(policy=policy):
                schedule = execute_adjusted_plan(attacked, plan, policy).set_index('ship')
                # 計画開始6時のV2を到着した3時に開始し、未到着のV3を待たない。
                self.assertEqual(schedule.loc['V2', 'start_time'], 3)
                self.assertEqual(schedule.loc['V3', 'start_time'], 6)
                np.testing.assert_array_equal(schedule.waiting_time, [0, 0, 0])
                # 遅い申告でも2時間を超える前倒しを許し、到着時点で開始する。
                late = apply_attack(data, {'V2': 8})
                late_schedule = execute_adjusted_plan(late, create_plan(late), policy).set_index('ship')
                self.assertEqual(late_schedule.loc['V2', 'planned_start_time'], 11)
                self.assertEqual(late_schedule.loc['V2', 'start_time'], 3)

    def test_reassignment_uses_free_berth_and_its_handling_time(self):
        data = small_data([0, 0, 1], [4, 1, 1])
        data['handling_time_B2'] = [10., 3., 10.]
        attacked = apply_attack(data, {'V2': 4})
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
        attacked = apply_attack(data, {'V1': 8})
        plan = create_plan(attacked)
        fixed = execute_fixed_plan(plan).set_index('ship')
        self.assertEqual(fixed.loc['V2', 'waiting_time'], 0)
        for policy in ('same_berth', 'reassign'):
            schedule = execute_adjusted_plan(attacked, plan, policy).set_index('ship')
            self.assertEqual(schedule.loc['V1', 'start_time'], 0)
            self.assertEqual(schedule.loc['V1', 'completion_time'], 10)
            self.assertEqual(schedule.loc['V2', 'start_time'], 10)
            self.assertEqual(schedule.loc['V2', 'waiting_time'], 6)

    def test_reassignment_can_propagate_a_false_report_to_other_ships(self):
        data = small_data([0, 2, 3], [3, 3, 3])
        normal = execute_adjusted_plan(data, create_plan(data), 'reassign').set_index('ship')
        attacked = apply_attack(data, {'V3': -4})
        schedule = execute_adjusted_plan(attacked, create_plan(attacked), 'reassign').set_index('ship')
        np.testing.assert_array_equal(normal.loc[['V1', 'V2', 'V3'], 'waiting_time'], [0, 1, 3])
        np.testing.assert_array_equal(schedule.loc[['V1', 'V2', 'V3'], 'waiting_time'], [0, 4, 0])
        np.testing.assert_array_equal(schedule.loc[['V1', 'V2', 'V3'], 'actual_arrival_time'], [0, 2, 3])
        self.assertEqual((schedule.waiting_time - normal.waiting_time).loc[['V1', 'V2']].mean(), 1.5)

    def test_all_cases_preserve_physics_inputs_and_paired_metrics(self):
        results = run_experiment(trials=2, duration_hours=None, num_ships=25, mean_interarrival_time=0.3, keep_schedules=True)
        data = results['ship_data'].set_index(['trial', 'ship'])
        schedules = results['schedules']
        keys = ['experiment_group', 'trial', 'attack_scale', 'shift_hours', 'scenario', 'policy']
        for _, case in schedules.groupby(keys):
            self.assertEqual(case.ship.nunique(), 25)
            self.assertTrue((case.start_time >= case.actual_arrival_time).all())
            if case.policy.iloc[0] == 'fixed':
                self.assertTrue((case.start_time >= case.planned_start_time).all())
            np.testing.assert_allclose(case.turnaround_time, case.waiting_time + case.handling_time)
            for berth, rows in case.groupby('assigned_berth'):
                rows = rows.sort_values('start_time')
                self.assertTrue((rows.start_time.to_numpy()[1:] >= rows.completion_time.to_numpy()[:-1]).all())
                expected = [data.loc[(r.trial, r.ship), f'handling_time_{berth}'] for r in rows.itertuples()]
                np.testing.assert_allclose(rows.handling_time, expected)
            for row in case.itertuples():
                self.assertEqual(row.actual_arrival_time, data.loc[(row.trial, row.ship), 'actual_arrival_time'])
        normal = schedules[schedules.scenario == 'normal']
        self.assertTrue((normal.waiting_time >= 0).all())
        self.assertTrue((normal.waiting_time > 0).any())
        compare = results['summary']
        for row in compare.itertuples():
            runs = results['run_metrics']
            selected = runs[(runs.experiment_group == row.experiment_group)
                            & (runs.attack_scale == row.attack_scale) & (runs.shift_hours == row.shift_hours)
                            & (runs.scenario == row.scenario) & (runs.policy == row.policy)]
            self.assertAlmostEqual(row.mean, selected[row.metric].mean())
            baseline = runs[(runs.attack_scale == row.attack_scale)
                            & (runs.scenario == 'normal') & (runs.policy == row.policy)].set_index('trial')
            differences = selected.set_index('trial')[row.metric] - baseline[row.metric]
            self.assertAlmostEqual(row.normal_mean, baseline[row.metric].mean())
            self.assertAlmostEqual(row.mean_difference, differences.mean())
            self.assertAlmostEqual(row.std_difference, differences.std(ddof=1))

    def test_zero_shift_and_repeated_seed_reproduce_results(self):
        settings = dict(trials=1, duration_hours=None, num_ships=10, shift_hours=0)
        result = run_experiment(**settings)
        repeated = run_experiment(**settings)
        for name in result:
            pd.testing.assert_frame_equal(result[name], repeated[name])
        np.testing.assert_allclose(result['summary'].mean_difference, 0)
        self.assertTrue((result['summary'].unchanged_trials == 1).all())

    def test_full_study_counts_targets_and_three_policies_share_inputs(self):
        result = run_experiment(trials=1, full_study=True, keep_schedules=True)
        runs = result['run_metrics']
        self.assertEqual(len(result['plan_diagnostics']), 1)
        self.assertEqual(POLICIES, ('fixed', 'same_berth', 'reassign'))
        self.assertEqual(len(runs.query("scenario!='normal'")), 99)
        self.assertEqual(len(runs.query("experiment_group=='main'")), 72)
        self.assertEqual(len(runs.query("experiment_group=='mixed_supplement'")), 27)
        self.assertEqual(len(runs.query("scenario=='normal'")), 12)
        self.assertEqual(runs.seed.unique().tolist(), [42])
        self.assertNotIn('demand_multiplier', runs)
        self.assertNotIn('target_position', runs)
        self.assertTrue((result['summary'].trials == 1).all())
        self.assertTrue(result['summary'].std_difference.isna().all())
        self.assertTrue(runs.query("attack_scale=='single' and scenario=='mixed'").empty)
        schedules = result['schedules']
        self.assertTrue((schedules.start_time >= schedules.actual_arrival_time).all())
        keys = ['attack_scale','shift_hours','scenario','policy']
        for _, case in schedules.groupby(keys):
            self.assertEqual(case.ship.nunique(), len(case))
            for _, rows in case.groupby('assigned_berth'):
                rows = rows.sort_values('start_time')
                self.assertTrue((rows.start_time.to_numpy()[1:] >= rows.completion_time.to_numpy()[:-1]).all())
            if case.scenario.iloc[0] != 'normal':
                changes = json.loads(case.report_shifts.iloc[0])
                actual = case.set_index('ship').actual_arrival_time
                reported = case.set_index('ship').reported_arrival_time
                for ship in case.ship:
                    self.assertAlmostEqual(reported[ship]-actual[ship], changes.get(ship,0))
        for _, rows in runs.query("scenario!='normal'").groupby(['attack_scale','shift_hours','scenario']):
            self.assertEqual(rows.target_ships.nunique(),1)
            self.assertEqual(rows.report_shifts.nunique(),1)
            self.assertEqual(rows.target_count.nunique(),1)

    def test_target_sets_are_nested_reproducible_and_mixed_signs_are_balanced(self):
        data = generate_ship_data(num_ships=56)
        selected, directions = select_targets(data, True, 'V5', 42)
        self.assertEqual((selected,directions),select_targets(data,True,'V5',42))
        self.assertEqual([len(ships) for _,ships in selected],[1,6,17,28])
        previous = set()
        for _,ships in selected:
            self.assertEqual(len(set(ships)),len(ships))
            self.assertTrue(previous.issubset(ships))
            self.assertLessEqual(abs(sum(directions[ship] for ship in ships)),1)
            previous=set(ships)
        other,_=select_targets(data,True,'V5',43)
        self.assertNotEqual(selected,other)

    def test_multi_attack_preserves_physical_inputs_and_zero_targets(self):
        data=small_data([0,2,3],[3,3,3])
        saved=data.copy(deep=True)
        attacked=apply_attack(data,{'V1':4,'V3':-4})
        np.testing.assert_array_equal(attacked.reported_arrival_time,[4,2,0])
        pd.testing.assert_frame_equal(data,saved)
        pd.testing.assert_frame_equal(attacked.drop(columns='reported_arrival_time'),data.drop(columns='reported_arrival_time'))
        pd.testing.assert_frame_equal(apply_attack(data,{}),data)
        with self.assertRaises(ValueError):apply_attack(data,{'missing':4})
        with self.assertRaises(ValueError):apply_attack(data,{'V1':float('nan')})

    def test_group_metrics_and_increased_ship_counts_match_hand_calculation(self):
        data=small_data([0,2,3],[3,3,3])
        normal=execute_fixed_plan(create_plan(data))
        changed=apply_attack(data,{'V3':-4})
        schedule=execute_adjusted_plan(changed,create_plan(changed),'reassign')
        metrics=calculate_metrics(schedule,['V1','V3'],normal)
        self.assertEqual(metrics['target_average_waiting_time'],0)
        self.assertEqual(metrics['other_average_waiting_time'],4)
        self.assertEqual(metrics['increased_waiting_ships'],1)
        self.assertEqual(metrics['other_increased_waiting_ships'],1)
        empty=calculate_metrics(normal,[])
        self.assertTrue(np.isnan(empty['target_average_waiting_time']))
        all_target=calculate_metrics(normal,data.ship.tolist())
        self.assertTrue(np.isnan(all_target['other_average_waiting_time']))

    def test_queue_profile_merges_simultaneous_arrivals_and_starts(self):
        from plotting import queue_profile
        data=small_data([0,2,3],[3,3,3])
        schedule=execute_fixed_plan(create_plan(data))
        times,counts=queue_profile(schedule,9)
        np.testing.assert_array_equal(times,[0,2,3,6,9])
        np.testing.assert_array_equal(counts,[0,1,1,0,0])

    def test_default_storage_keeps_only_the_declared_example(self):
        compact=run_experiment(trials=2,full_study=True)
        full=run_experiment(trials=2,full_study=True,keep_schedules=True)
        for name in ['ship_data','run_metrics','summary','plan_diagnostics']:
            pd.testing.assert_frame_equal(compact[name],full[name])
        sample=compact['schedules']
        self.assertEqual(sample.trial.nunique(),1)
        self.assertNotIn('demand_multiplier', sample)
        attacked=sample.query("scenario!='normal'")
        self.assertEqual(set(attacked.attack_scale),{'p30'})
        self.assertEqual(set(attacked.scenario),{'mixed'})
        self.assertLess(len(sample),len(full['schedules']))

    def test_ships_finishing_after_arrival_period_are_not_dropped(self):
        result = run_experiment(trials=1, duration_hours=1, mean_interarrival_time=0.1, num_berths=1, target_ship='V2')
        data = result['ship_data']
        self.assertTrue((data.actual_arrival_time < 1).all())
        self.assertGreater(result['plan_diagnostics'].last_planned_completion.iloc[0], 1)
        self.assertNotIn('candidate_arrival_time', data.columns)
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

    def test_saved_output_contains_three_japanese_sheets_and_preserves_values(self):
        args = create_parser().parse_args(['--trials', '1', '--no-plot'])
        results = run_experiment(trials=1)
        config = create_config(args, results, 3.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_results(results, config, output)
            self.assertEqual({path.name for path in output.iterdir()},
                             {'config.json', 'results.xlsx'})
            with pd.ExcelFile(output / 'results.xlsx') as workbook_file:
                self.assertEqual(workbook_file.sheet_names, ['条件別集計', '試行別結果', '船別データ'])
            from openpyxl import load_workbook
            workbook = load_workbook(output / 'results.xlsx')
            for sheet in workbook:
                if sheet.title == '条件別集計':
                    self.assertEqual(sheet.freeze_panes, 'E8')
                    self.assertEqual(sheet.tables['WaitingOverview'].ref, 'A7:G13')
                    self.assertEqual(sheet.auto_filter.ref, 'A20:R98')
                    self.assertTrue(sheet['A17'].value.startswith('表2'))
                    self.assertEqual(sheet['G6'].hyperlink.target, "#'条件別集計'!A17")
                    self.assertEqual(sheet['G17'].hyperlink.target, "#'条件別集計'!A1")
                    continue
                self.assertEqual(sheet.freeze_panes, 'B2')
                self.assertEqual(sheet.auto_filter.ref, sheet.dimensions)
                self.assertFalse(sheet.sheet_view.showGridLines)
                headers = {cell.value: cell.column for cell in sheet[1]}
                for label in ['申告変更幅（時間）', '試行数', '試行番号', '改ざん対象隻数', '全船数（隻）']:
                    if label in headers:
                        self.assertTrue(all(cell[0].number_format == '0' for cell in
                                            sheet.iter_rows(min_row=2, min_col=headers[label], max_col=headers[label])))
            ships_sheet = workbook['船別データ']
            self.assertEqual(ships_sheet['B3'].number_format, '0.#')
            for value, expected in [(2, '0'), (1.1, '0.##'), (161.609, '0'),
                                    (0.0000227928, '0.#######'), (99.999, '0'),
                                    (-0.0012345, '0.#####'), (1e-15, '0')]:
                self.assertEqual(excel_number_format(value), expected)
            self.assertEqual(excel_number_format(1 / 63, percentage=True), '0.##%')
            workbook.close()
            saved_overview = pd.read_excel(output / 'results.xlsx', sheet_name='条件別集計', header=6, nrows=6)
            expected = translate_excel_table(create_summary_overview(results['summary']), SUMMARY_OVERVIEW_COLUMNS)
            pd.testing.assert_frame_equal(saved_overview, expected, check_dtype=False)
            self.assertEqual(saved_overview['運用方式'].tolist(), ['固定', '同じバース内調整', 'バース変更'] * 2)
            saved_ships = pd.read_excel(output / 'results.xlsx', sheet_name='船別データ')
            self.assertEqual(saved_ships['船ID'].nunique(), len(results['ship_data']))
            self.assertIn('実到着時刻（時間）', saved_ships)
            self.assertIn('正常時の申告到着時刻（時間）', saved_ships)
            for berth in range(1, 5):
                self.assertIn(f'B{berth}の荷役時間（時間）', saved_ships)
            restored = saved_ships.rename(columns={label: key for key, label in EXCEL_COLUMNS.items()})
            pd.testing.assert_frame_equal(restored, results['ship_data'], check_dtype=False)
            for name, sheet_name in [('run_metrics', '試行別結果'), ('summary', '条件別集計')]:
                header = 19 if name == 'summary' else 0
                saved = pd.read_excel(output / 'results.xlsx', sheet_name=sheet_name, header=header)
                self.assertIn('運用方式', saved)
                self.assertEqual(set(saved['運用方式']), {'固定', '同じバース内調整', 'バース変更'})
                self.assertIn('早い申告', saved['申告条件'].values)
                self.assertIn('遅い申告', saved['申告条件'].values)
                if name == 'summary':
                    self.assertIn('全船の平均待ち時間（時間）', saved['評価指標'].values)
                else:
                    self.assertEqual(saved['改ざん対象船ID'].iloc[0], '["V5"]')
                    self.assertTrue(pd.api.types.is_numeric_dtype(saved['申告変更幅（時間）']))
                saved = saved.rename(columns={label: key for key, label in EXCEL_COLUMNS.items()})
                for column, labels in EXCEL_VALUES.items():
                    if column in saved:
                        saved[column] = saved[column].replace({label: key for key, label in labels.items()})
                pd.testing.assert_frame_equal(saved, results[name], check_dtype=False)
            saved_config = json.loads((output / 'config.json').read_text())
            self.assertEqual(saved_config['first_seed'], 42)
            self.assertEqual(saved_config['policies'], ['fixed', 'same_berth', 'reassign'])
            self.assertIn('normal_plan_diagnostics', saved_config)
            self.assertEqual(saved_config['model_version'], 'discrete_4berth_multi_ship_v6')
            self.assertEqual(len(saved_config['normal_waiting_summary']), 3)
            self.assertNotIn('max_arrival_adjustment_hours', saved_config['normal_plan_diagnostics'])
            self.assertIn('mean_difference', results['summary'])
            self.assertNotIn('demand_multipliers', saved_config)
            self.assertNotIn('demand_multiplier', results['run_metrics'])
            self.assertNotIn('target_position', results['run_metrics'])


if __name__ == '__main__':
    unittest.main()
