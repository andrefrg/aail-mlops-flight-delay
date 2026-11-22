"""Experiments must respect temporal boundaries and the raw input contract."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from experiments.compare_models import build_model, compare, development_sample, validate_protocol
from flight_delay.data.audit import BASE_FEATURES
from flight_delay.data.prepare import prepare
from test_baseline import fixture


class ExperimentTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "configs/baseline.json").read_text())

    def test_sampling_excludes_holdout_and_ignores_labels(self):
        config = self.config()
        frame = pd.concat([fixture(2022), fixture(2023)], ignore_index=True)
        frame["row_id"] = range(len(frame))
        frame["month"] = frame.FlightDate.str[:7]
        frame.loc[35, "FlightDate"] = "2023-07-01"
        selected = development_sample(frame, config, 5)
        frame["target"] = "severe_delay"
        self.assertEqual(
            selected.row_id.tolist(), development_sample(frame, config, 5).row_id.tolist()
        )
        self.assertNotIn(35, selected.row_id.tolist())
        self.assertTrue(selected.groupby("month").size().le(5).all())
        config["folds"][0]["validation_end"] = "2023-08-01"
        with self.assertRaises(ValueError):
            validate_protocol(config)

    def test_candidates_accept_unknown_categories(self):
        frame = fixture()
        future = frame[BASE_FEATURES].iloc[:2].copy()
        future["Origin"] = "UNKNOWN"
        for name in ["classroom_logistic", "balanced_logistic", "balanced_forest"]:
            model = build_model(name).fit(frame[BASE_FEATURES], frame.target)
            self.assertEqual(len(model.predict(future)), 2)
        self.assertEqual(
            build_model("classroom_logistic")[-1].get_params(),
            LogisticRegression(C=0.1, max_iter=200).get_params(),
        )

    def test_comparison_without_holdout_partitions(self):
        config = self.config()
        config.update(sample_per_month=100, chunksize=7)
        config["folds"] = [
            {"name": "validation", "validation_start": "2023-01-01", "validation_end": "2023-07-01"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            for year in [2022, 2023]:
                frame = fixture(year)
                if year == 2023:
                    frame.loc[12:, "FlightDate"] = [f"2023-07-{day:02d}" for day in range(1, 7)]
                frame.to_csv(raw / f"flights_{year}.csv", index=False)
            settings = root / "config.json"
            settings.write_text(json.dumps(config))
            data = root / "processed"
            prepare(raw, data, settings)
            for path in (data / "monthly").glob("month=*"):
                if path.name.split("=", 1)[1] >= "2023-07":
                    shutil.rmtree(path)
            summary = compare(data, root / "results", config, 100)
            self.assertEqual(
                set(summary.index),
                {"majority", "classroom_logistic", "balanced_logistic", "balanced_forest"},
            )
            scores = pd.read_csv(root / "results/fold_metrics.csv")
            self.assertEqual(scores.validation_rows.tolist(), [12] * 4)
            self.assertEqual(scores.train_rows.nunique(), 1)
            self.assertFalse(
                json.loads((root / "results/selection.json").read_text())["final_test_evaluated"]
            )
            with self.assertRaises(FileExistsError):
                compare(data, root / "results", config, 100)
