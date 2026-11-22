"""Small end-to-end model-only training; no services or tracking dependencies."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from flight_delay.models.tracking import tracking_metrics

from flight_delay.data.prepare import prepare
from flight_delay.models import training
from test_baseline import fixture
from train import register_model


class WorkflowTests(unittest.TestCase):
    def test_commit_alias_points_to_registered_version(self):
        mlflow = Mock()
        mlflow.register_model.return_value = SimpleNamespace(version="7")
        sha = "a" * 40

        register_model(mlflow, "models:/example", "model", sha)

        mlflow.MlflowClient.return_value.set_registered_model_alias.assert_called_once_with(
            "model", sha, "7"
        )

    def test_invalid_commit_is_not_registered(self):
        mlflow = Mock()
        with self.assertRaises(ValueError):
            register_model(mlflow, "models:/example", "model", "invalid")
        mlflow.register_model.assert_not_called()

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


class AutomaticSelectionTests(unittest.TestCase):
    def test_macro_f1_selects_different_candidate_than_accuracy(self):
        rows = pd.DataFrame(
            [
                {"model": "majority", "accuracy": 0.99, "macro_f1": 0.99},
                {"model": "linear", "accuracy": 0.8, "macro_f1": 0.3, "test_macro_f1": 0.99},
                {"model": "forest", "accuracy": 0.5, "macro_f1": 0.4, "test_macro_f1": 0.1},
            ]
        )
        variants = [{"name": "linear"}, {"name": "forest"}]
        selected, _ = training.choose_variant(rows, variants, metric="macro_f1")
        self.assertEqual(selected["name"], "forest")
        rows.loc[rows.model.eq("linear"), "macro_f1"] = 0.4
        selected, _ = training.choose_variant(rows, variants, metric="macro_f1")
        self.assertEqual(selected["name"], "linear")

    def test_incomplete_or_nonfinite_scores_are_rejected(self):
        variants = [{"name": "linear"}, {"name": "forest"}]
        rows = pd.DataFrame(
            [
                {"model": "linear", "fold": "one", "macro_f1": 0.4},
                {"model": "linear", "fold": "two", "macro_f1": 0.5},
                {"model": "forest", "fold": "one", "macro_f1": 0.6},
            ]
        )
        with self.assertRaises(ValueError):
            training.choose_variant(rows, variants, "macro_f1", expected_folds=2)
        rows.loc[0, "macro_f1"] = float("nan")
        with self.assertRaises(ValueError):
            training.choose_variant(rows, variants, "macro_f1")

    def test_random_forest_winner_is_exported_and_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for year in [2022, 2023]:
                frame = fixture(year)
                if year == 2023:
                    frame.loc[12:, "FlightDate"] = [f"2023-07-{day:02d}" for day in range(1, 7)]
                frame.to_csv(raw / f"flights_{year}.csv", index=False)
            config = json.loads(
                (Path(__file__).resolve().parents[1] / "configs/baseline.json").read_text()
            )
            config.update(sample_per_month=100, chunksize=7)
            config["folds"] = [
                {
                    "name": "validation",
                    "validation_start": "2023-01-01",
                    "validation_end": "2023-07-01",
                }
            ]
            config["variants"] = [
                {
                    "name": "forest",
                    "estimator": "RandomForestClassifier",
                    "n_estimators": 5,
                    "min_samples_leaf": 1,
                    "n_jobs": 1,
                }
            ]
            settings = root / "config.json"
            settings.write_text(json.dumps(config))
            data = root / "data/processed"
            prepare(raw, data, settings)
            reports = root / "reports/generated/baseline"
            with patch.multiple(
                training, ROOT=root, DATA=data, MODELS=root / "models", REPORTS=reports
            ):
                with patch.object(training, "code_fingerprint", return_value="test-code"):
                    with patch.object(training, "evaluate", wraps=training.evaluate) as evaluate:
                        selection = training.train(settings)
                        for call in evaluate.call_args_list:
                            self.assertLessEqual(call.args[3], config["holdout_start"])
                    results = training.evaluate_model(settings)
            model = joblib.load(root / "models/model.joblib")
            self.assertIsInstance(model[-1], RandomForestClassifier)
            self.assertEqual(selection["estimator"], "RandomForestClassifier")
            meta = json.loads((root / "models/model_metadata.json").read_text())
            self.assertIsNone(meta["C"])
            logged = tracking_metrics(reports, selection, results["baseline"])
            self.assertIn("cv_macro_f1_mean", logged)
            self.assertEqual(logged["test_macro_f1"], results["baseline"]["macro_f1"])
            self.assertTrue((reports / "validation_details.json").is_file())

    def test_tracking_keeps_validation_and_test_metrics_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            pd.DataFrame(
                [
                    {"model": "forest", "accuracy": 0.5, "macro_f1": 0.4},
                    {"model": "forest", "accuracy": 0.7, "macro_f1": 0.6},
                ]
            ).to_csv(reports / "cross_validation.csv", index=False)
            metrics = tracking_metrics(reports, {"variant": {"name": "forest"}})
            self.assertAlmostEqual(metrics["cv_macro_f1_mean"], 0.5)
            self.assertAlmostEqual(metrics["cv_macro_f1_std"], 0.1)
            self.assertNotIn("test_macro_f1", metrics)


if __name__ == "__main__":
    unittest.main()
