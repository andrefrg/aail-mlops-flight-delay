"""Small end-to-end model-only training; no services or tracking dependencies."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from flight_delay.data.prepare import prepare
from flight_delay.models import training
from test_baseline import fixture


class WorkflowTests(unittest.TestCase):
    def test_selection_uses_accuracy_and_excludes_majority(self):
        rows = pd.DataFrame(
            [
                {"model": "majority", "accuracy": 0.99, "macro_f1": 0.3},
                {"model": "logistic_C0.1", "accuracy": 0.80, "macro_f1": 0.4},
                {"model": "logistic_C1.0", "accuracy": 0.81, "macro_f1": 0.35},
                {"model": "logistic_C10.0", "accuracy": 0.79, "macro_f1": 0.5},
            ]
        )
        variants = [{"name": f"logistic_C{C}", "C": C} for C in [0.1, 1.0, 10.0]]
        selected, _ = training.choose_variant(rows, variants)
        self.assertEqual(selected["C"], 1.0)

    def test_prepare_train_evaluate_export_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (root / "docs").mkdir()
            for year in [2022, 2023]:
                data = fixture(year)
                if year == 2023:
                    data.loc[data.index >= 12, "FlightDate"] = [
                        f"2023-07-{day:02d}" for day in range(1, 7)
                    ]
                data.to_csv(raw / f"flights_{year}.csv", index=False)
            config = json.loads(
                (Path(__file__).resolve().parents[1] / "configs/baseline.json").read_text()
            )
            config["sample_per_month"] = 100
            config["chunksize"] = 7
            config["folds"] = [
                {
                    "name": "validation",
                    "validation_start": "2023-01-01",
                    "validation_end": "2023-07-01",
                }
            ]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            data = root / "data/processed"
            prepare(raw, data, config_path)
            models = root / "models"
            reports = root / "reports/generated/baseline"
            with (
                patch.multiple(training, ROOT=root, DATA=data, MODELS=models, REPORTS=reports),
                patch.object(training, "code_fingerprint", return_value="test-code"),
            ):
                selection = training.train(config_path)
                results = training.evaluate_model(config_path)
                self.assertEqual(
                    training.train(config_path)["model_sha256"], selection["model_sha256"]
                )
                self.assertEqual(training.evaluate_model(config_path), results)
            model = joblib.load(models / "model.joblib")
            self.assertIsInstance(model[-1], LogisticRegression)
            self.assertEqual(model[-1].max_iter, 200)
            metadata = json.loads((models / "model_metadata.json").read_text())
            self.assertEqual(metadata["name"], "model")
            self.assertNotIn("registry", metadata)
            self.assertEqual(results["baseline"]["rows"], 6)
            self.assertTrue((root / "reports/tables/temporal_splits.csv").exists())


if __name__ == "__main__":
    unittest.main()
