"""Experiments must respect temporal boundaries and the raw input contract."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from experiments.compare_models import build_model, compare, development_sample, validate_protocol
from flight_delay.data.audit import BASE_FEATURES
from flight_delay.data.prepare import prepare
from flight_delay.models.pipeline import make_pipeline
from flight_delay.models.tracking import log_selected_model
from scripts.app import create_app, load_model
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

    def test_custom_candidate_parameters_are_used(self):
        for estimator in ["RandomForestClassifier", "ExtraTreesClassifier"]:
            variant = {"name": "custom", "estimator": estimator, "n_estimators": 3,
                       "max_depth": 4, "min_samples_leaf": 2, "n_jobs": 1}
            model = build_model("custom", seed=17, variant=variant)[-1]
            self.assertEqual(model.n_estimators, 3)
            self.assertEqual(model.max_depth, 4)
            self.assertEqual(model.random_state, 17)

    def test_new_estimators_support_skops_roundtrip(self):
        import skops.io as sio

        frame = fixture()
        for estimator in ["ExtraTreesClassifier", "LinearSVC"]:
            with self.subTest(estimator=estimator):
                variant = {"name": "new", "estimator": estimator, "n_estimators": 3,
                           "min_samples_leaf": 1, "n_jobs": 1}
                model = make_pipeline(variant, {"seed": 42}).fit(frame[BASE_FEATURES], frame.target)
                saved = sio.dumps(model)
                trusted = ["flight_delay.features.build.FlightFeatures", "numpy.dtype"]
                if estimator == "ExtraTreesClassifier":
                    trusted.append("sklearn.tree._tree.Tree")
                restored = sio.loads(saved, trusted=trusted)
                np.testing.assert_array_equal(
                    restored.predict(frame[BASE_FEATURES]), model.predict(frame[BASE_FEATURES])
                )

    def test_candidates_accept_unknown_categories(self):
        frame = fixture()
        future = frame[BASE_FEATURES].iloc[:2].copy()
        future["Origin"] = "UNKNOWN"
        for variant in self.config()["variants"]:
            with self.subTest(candidate=variant["name"]):
                model = build_model(variant["name"]).fit(frame[BASE_FEATURES], frame.target)
                predicted = model.predict(future)
                self.assertEqual(len(predicted), 2)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "model.joblib"
                    joblib.dump(model, path)
                    np.testing.assert_array_equal(joblib.load(path).predict(future), predicted)
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
                {"majority", *[v["name"] for v in config["variants"]]},
            )
            scores = pd.read_csv(root / "results/fold_metrics.csv")
            self.assertEqual(scores.validation_rows.tolist(), [12] * (1 + len(config["variants"])))
            self.assertEqual(scores.train_rows.nunique(), 1)
            self.assertFalse(
                json.loads((root / "results/selection.json").read_text())["final_test_evaluated"]
            )
            with self.assertRaises(FileExistsError):
                compare(data, root / "results", config, 100)


def test_catboost_and_mlp_load_from_registry_alias(tmp_path):
    import mlflow
    import mlflow.sklearn

    frame = fixture()
    future = frame[BASE_FEATURES].iloc[:5].copy()
    future["Origin"] = "UNKNOWN"
    future["Distance"] = np.nan
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        client = mlflow.MlflowClient()
        artifact_root = (tmp_path / "artifacts").as_uri()
        experiment_id = client.create_experiment("local-validation", artifact_root)
        config = json.loads((ROOT / "configs/baseline.json").read_text())
        variants = {variant["name"]: variant for variant in config["variants"]}
        for name in ["engineered_catboost", "baseline_torch_mlp"]:
            candidate = make_pipeline(variants[name], config).fit(frame[BASE_FEATURES], frame.target)
            with mlflow.start_run(experiment_id=experiment_id):
                info = log_selected_model(mlflow, candidate, ROOT / "src/flight_delay")
            registered = mlflow.register_model(info.model_uri, "local-voos-model")
            client.set_registered_model_alias("local-voos-model", "staging", registered.version)
            with patch.dict(
                "os.environ", MLFLOW_MODEL_NAME="local-voos-model", MODEL_ALIAS="staging"
            ):
                restored = load_model()
            response = create_app(model=restored).test_client().post(
                "/classify", json={"instances": json.loads(future.to_json(orient="records"))}
            )
            assert response.status_code == 200
            assert response.get_json()["classifications"] == candidate.predict(future).tolist()
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)
