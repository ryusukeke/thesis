"""モデルの境界、手計算、比較入力の共有と保存内容を検証する。"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import openpyxl
import pandas as pd
from pandas.testing import assert_frame_equal

from main import create_parser, make_settings
from result_output import ROOT, make_config, save_results
from simulation import (
    build_conditions, choose_next_ship, generate_input, mark_long_handling,
    measure_run, run_experiment, simulate,
)


def ships(arrivals, handling, names=None):
    names = names or [f"V{i:04d}" for i in range(len(arrivals))]
    return pd.DataFrame({"ship": names, "arrival": arrivals, "handling": handling})


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.example = ships([0.0] * 4, [12.0, 8.0, 2.0, 2.0], list("ABCD"))

    def test_hand_calculation_and_paired_metrics(self):
        fifo = simulate(self.example, "fifo", outage_start=0)["schedule"]
        spt = simulate(self.example, "spt", outage_start=0)["schedule"]
        normal = simulate(self.example, "normal")["schedule"]
        self.assertEqual(fifo.start.tolist(), [2, 2, 10, 12])
        self.assertEqual(spt.start.tolist(), [4, 4, 2, 2])
        self.assertEqual(fifo.waiting.mean(), 6.5)
        self.assertEqual(spt.waiting.mean(), 3)
        condition = {"method": "spt", "duration": 24, "preparation": 2}
        metrics = measure_run(spt, normal.sample(frac=1), fifo.sample(frac=1), condition, 0)
        self.assertEqual(metrics["fifo_delta_wait"], -3.5)
        self.assertEqual(metrics["fifo_delta_long_wait"], 2)
        self.assertEqual(metrics["fifo_worse_ships"], 2)
        self.assertEqual(metrics["manual_starts"], 4)
        self.assertEqual(metrics["manual_berth_hours"], 24)
        self.assertEqual(metrics["cohort_ships"], 4)
        self.assertEqual(metrics["cohort_clearance"], 0)

    def test_existing_handling_can_exceed_manual_limit_without_interruption(self):
        data = ships([0, 0, 0, 0, 1, 1], [10, 12, 14, 16, 1, 1])
        result = simulate(data, "fifo", outage_start=1, duration=30, preparation=1, capacity=2)
        schedule = result["schedule"]
        self.assertEqual(schedule.start.tolist(), [0, 0, 0, 0, 14, 15])
        audit = result["events"].query("time == 2").iloc[0]
        self.assertEqual((audit.busy_before, audit.starts, audit.busy_after), (4, 0, 4))

    def test_simultaneous_completion_arrival_and_restoration(self):
        data = ships([0, 1, 3], [1, 1, 1])
        result = simulate(data, "fifo", outage_start=1, duration=2, preparation=2)
        self.assertEqual(result["schedule"].start.tolist(), [0, 3, 3])
        self.assertEqual(result["schedule"].phase.tolist(), ["通常", "復旧後", "復旧後"])
        self.assertEqual(len(result["events"].query("time == 3")), 1)

    def test_preparation_completion_allows_immediate_start(self):
        data = ships([1, 2], [1, 1])
        result = simulate(data, "fifo", outage_start=1, duration=10, preparation=1)
        self.assertEqual(result["schedule"].start.tolist(), [2, 2])
        self.assertEqual(result["schedule"].phase.tolist(), ["手動", "手動"])

    def test_preparation_at_least_duration_or_zero_capacity_matches_wait(self):
        wait = simulate(self.example, "wait", outage_start=0)["schedule"]
        for preparation, capacity in ((24, 2), (30, 2), (0, 0)):
            actual = simulate(self.example, "fifo", outage_start=0,
                              preparation=preparation, capacity=capacity)["schedule"]
            assert_frame_equal(actual, wait)

    def test_no_effect_extreme_conditions(self):
        data = generate_input(42, 3.0)
        normal = simulate(data, "normal")["schedule"].drop(columns="phase")
        for options in ({"duration": 0}, {"outage_start": 10000},
                        {"preparation": 0, "capacity": 4}):
            actual = simulate(data, "fifo", **options)["schedule"].drop(columns="phase")
            assert_frame_equal(actual, normal)

    def test_equal_service_times_give_same_manual_order(self):
        data = ships([0] * 9, [3] * 9)
        fifo = simulate(data, "fifo", outage_start=0)["schedule"]
        spt = simulate(data, "spt", outage_start=0)["schedule"]
        assert_frame_equal(fifo, spt)

    def test_selector_receives_only_arrived_ships(self):
        data = ships([0, 0, 5], [10, 10, 1], list("ABC"))
        seen = []

        def observe(waiting, short_first):
            seen.append({s["ship"] for s in waiting})
            return choose_next_ship(waiting, short_first)

        with patch("simulation.choose_next_ship", side_effect=observe):
            result = simulate(data, "spt", outage_start=0, duration=30,
                              preparation=0, capacity=1)
        self.assertEqual(seen[0], {"A", "B"})
        self.assertEqual(result["schedule"].start.tolist(), [0, 11, 10])

    def test_long_group_is_fixed_by_input_and_ties(self):
        data = mark_long_handling(ships([0] * 5, [4, 4, 3, 2, 1], list("BACDE")))
        self.assertEqual(set(data.loc[data.long_handling, "ship"]), {"A", "B"})
        for method in ("normal", "wait", "fifo", "spt"):
            output = simulate(data, method)["schedule"]
            self.assertEqual(set(output.loc[output.long_handling, "ship"]), {"A", "B"})

    def test_stopped_arrival_cohort_excludes_restoration_boundary(self):
        data = ships([47, 48, 71, 72], [2] * 4)
        schedule = simulate(data, "wait")["schedule"]
        normal = simulate(data, "normal")["schedule"]
        condition = {"method": "wait", "duration": 24, "preparation": 0}
        metrics = measure_run(schedule, normal, None, condition, 48)
        self.assertEqual(metrics["cohort_ships"], 2)
        self.assertEqual(metrics["cohort_mean_wait"], 12.5)
        self.assertEqual(metrics["cohort_clearance"], 0)

    def test_physical_constraints_and_all_ships_complete(self):
        data = generate_input(45, 1.5)
        for method in ("normal", "wait", "fifo", "spt"):
            result = simulate(data, method, duration=48)
            schedule = result["schedule"]
            self.assertEqual(set(data.ship), set(schedule.ship))
            self.assertFalse(schedule.ship.duplicated().any())
            self.assertTrue((schedule.start >= schedule.arrival).all())
            np.testing.assert_allclose(schedule.completion - schedule.start, schedule.handling)
            for _, group in schedule.groupby("berth"):
                group = group.sort_values("start")
                self.assertTrue((group.start.to_numpy()[1:] >= group.completion.to_numpy()[:-1]).all())
            audit = result["events"]
            manual_starts = audit[(audit.phase == "手動") & (audit.starts > 0)]
            self.assertTrue((manual_starts.busy_after <= 2).all())
            self.assertTrue((audit.loc[audit.phase.isin(["準備中", "復旧待ち"]), "starts"] == 0).all())

    def test_input_reproducibility_and_independent_handling_stream(self):
        a = generate_input(42, 3)
        b = generate_input(42, 1.5)
        assert_frame_equal(a, generate_input(42, 3))
        np.testing.assert_array_equal(a.handling, b.handling.iloc[:len(a)])
        self.assertTrue((a.arrival < 168).all())

    def test_empty_input_and_empty_cohort(self):
        data = ships([], [])
        self.assertTrue(simulate(data)["schedule"].empty)
        one = ships([0], [1])
        output = simulate(one)["schedule"]
        metrics = measure_run(output, output, output,
                              {"method": "fifo", "duration": 24, "preparation": 2}, 48)
        self.assertEqual(metrics["cohort_ships"], 0)
        self.assertEqual(metrics["cohort_clearance"], 0)
        self.assertTrue(np.isnan(metrics["cohort_mean_wait"]))

    def test_invalid_input_is_rejected(self):
        for field, value in (("arrival", -1), ("arrival", np.nan),
                             ("handling", 0), ("handling", np.inf)):
            data = self.example.copy()
            data.loc[0, field] = value
            with self.assertRaises(ValueError):
                simulate(data)
        for kwargs in ({"duration": -1}, {"preparation": np.nan}, {"capacity": 5},
                       {"capacity": 1.5}, {"method": "unknown"}):
            with self.assertRaises(ValueError):
                simulate(self.example, **kwargs)
        with self.assertRaises(ValueError):
            simulate(pd.concat([self.example, self.example]))
        with self.assertRaises(ValueError):
            generate_input(42, 0)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = make_settings("両方", 2, 3.029126213592233)
        cls.results = run_experiment(cls.settings)

    def test_condition_counts_and_shared_execution(self):
        self.assertEqual(len(build_conditions("主実験")), 13)
        auxiliary = build_conditions("補助実験")
        self.assertEqual(len(auxiliary), 12)
        self.assertEqual(len(build_conditions()), 21)
        manual = auxiliary[auxiliary.method.isin(["fifo", "spt"])]
        self.assertEqual(set(zip(manual.preparation, manual.capacity)),
                         {(0, 2), (2, 2), (6, 2), (2, 1), (2, 3)})
        self.assertEqual(set(manual.duration), {24})
        self.assertEqual(manual.groupby(["preparation", "capacity"]).size().tolist(), [2] * 5)
        runs = self.results["runs"]
        self.assertEqual(len(runs), 84)
        self.assertEqual(runs.include_main.sum(), 52)
        self.assertEqual(runs.include_auxiliary.sum(), 48)
        self.assertEqual((runs.include_main & runs.include_auxiliary).sum(), 16)
        self.assertFalse(runs.duplicated(["trial", "load", "condition"]).any())

    def test_input_and_ship_records_are_shared_once(self):
        inputs = self.results["inputs"]
        schedules = self.results["schedules"]
        self.assertFalse(inputs.duplicated(["trial", "load", "ship"]).any())
        for (trial, load), source in inputs.groupby(["trial", "load"]):
            selected = schedules[(schedules.trial == trial) & (schedules.load == load)]
            for _, group in selected.groupby("condition"):
                assert_frame_equal(source[["ship", "arrival", "handling"]].reset_index(drop=True),
                                   group[["ship", "arrival", "handling"]].reset_index(drop=True))
        self.assertEqual(set(self.results["events"].trial), {1})

    def test_summary_uses_sample_sd_of_paired_trials(self):
        runs = self.results["runs"]
        row = self.results["summary"].query("load == 2 and method == 'spt'").iloc[0]
        paired = runs[(runs.load == row.load) & (runs.condition == row.condition)]
        self.assertAlmostEqual(row.fifo_delta_wait, paired.fifo_delta_wait.mean())
        self.assertAlmostEqual(row.fifo_delta_wait_sd, paired.fifo_delta_wait.std(ddof=1))
        self.assertEqual(row.fifo_increases + row.fifo_decreases + row.fifo_unchanged, 2)

    def test_excel_metadata_and_overwrite_protection(self):
        config = make_config(self.settings, self.results)
        for name, digest in config["sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "結果"
            save_results(self.results, config, output)
            book = openpyxl.load_workbook(output / "実験結果.xlsx", read_only=True)
            self.assertEqual(book.sheetnames, ["条件別集計", "試行別結果", "船別入力", "船別実行", "代表試行イベント"])
            for sheet in book:
                header = next(sheet.iter_rows(values_only=True))
                self.assertTrue(all(any("\u3040" <= c <= "\u9fff" for c in label) for label in header))
                self.assertEqual(sum(1 for _ in sheet.iter_rows(min_row=2)), len(self.results[
                    {"条件別集計": "summary", "試行別結果": "runs", "船別入力": "inputs",
                     "船別実行": "schedules", "代表試行イベント": "events"}[sheet.title]]))
            book.close()
            saved = json.loads((output / "実験条件.json").read_text())
            self.assertEqual(saved["実行運用数"], 84)
            with self.assertRaises(ValueError):
                save_results(self.results, config, output)

    def test_cli_has_three_options_and_ten_trial_default(self):
        args = create_parser().parse_args([])
        self.assertEqual((args.experiment, args.trials, args.output), ("両方", 10, Path("results")))
        with self.assertRaises(ValueError):
            run_experiment(make_settings("両方", 0, 3))


if __name__ == "__main__":
    unittest.main()
