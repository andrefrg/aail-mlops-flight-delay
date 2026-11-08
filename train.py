"""Prepare, train, evaluate and register the flight-delay model in MLflow."""

import argparse
from contextlib import nullcontext
import csv
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, pstdev
import sys

import joblib
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from flight_delay.data.prepare import prepare
from flight_delay.data.audit import sha256
from flight_delay.models.training import train, evaluate_model


def recorded_evaluation(config_path):
    """Verify saved artifacts before explicitly publishing a historical training result."""
    selection = json.loads((ROOT / "models/selection.json").read_text())
    report = json.loads((ROOT / "reports/generated/baseline/holdout.json").read_text())
    model_hash = sha256(ROOT / "models/model.joblib")
    if model_hash != selection["model_sha256"] or model_hash != report["model_sha256"]:
        raise ValueError("Saved model and evaluation do not match; registration cancelled")
    context = selection["execution_context"]
    if context["config_sha256"] != sha256(config_path):
        raise ValueError("Configuration changed; registration cancelled")
    if context["data_manifest_sha256"] != sha256(ROOT / "data/processed/manifest.json"):
        raise ValueError("Data manifest changed; registration cancelled")
    return selection, report["models"]["baseline"]


def tracking_metrics(selection, result=None):
    """Separate selected-variant validation scores from final-test metrics."""
    with (ROOT / "reports/generated/baseline/cross_validation.csv").open(newline="") as stream:
        scores = [
            float(row["accuracy"])
            for row in csv.DictReader(stream)
            if row["model"] == selection["variant"]["name"]
        ]
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("Missing or invalid validation accuracies for the selected model")
    metrics = {
        "cv_accuracy_mean": mean(scores),
        "cv_accuracy_std": pstdev(scores),
    }
    if result is not None:
        metrics.update({
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "balanced_accuracy": result["balanced_accuracy"],
            "recall_severe_delay": result["classes"]["severe_delay"]["recall"],
            "recall_moderate_delay": result["classes"]["moderate_delay"]["recall"],
            "precision_severe_delay": result["classes"]["severe_delay"]["precision"],
        })
    return metrics


def register_model(mlflow, model_uri, model_name, commit_sha=None):
    """Register the logged model and identify its exact version by commit."""
    if commit_sha and not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise ValueError("COMMIT_SHA must be a full Git commit SHA")
    version = mlflow.register_model(model_uri, model_name)
    if commit_sha:
        mlflow.MlflowClient().set_registered_model_alias(model_name, commit_sha, version.version)
    return version


def main():
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["all", "prepare", "train", "evaluate", "register"], default="all"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument(
        "--no-tracking", action="store_true", help="Run locally without contacting MLflow"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Explicitly repeat training and overwrite local model/results",
    )
    args = parser.parse_args()
    if args.phase in ["all", "prepare"]:
        output = ROOT / "data/processed"
        if (output / "manifest.json").exists() and not args.force_prepare:
            print("Using existing prepared data.", flush=True)
        else:
            prepare(args.data_dir, output, args.config, force=args.force_prepare)
    if args.phase == "prepare":
        return

    mlflow = None
    if not args.no_tracking:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://10.17.0.253:8080"))
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "voo-classification"))

    with mlflow.start_run() if mlflow is not None else nullcontext():
        if args.phase == "register":
            selection, result = recorded_evaluation(args.config)
            print("Registering the saved model and recorded evaluation; no retraining.", flush=True)
            if mlflow is not None:
                mlflow.set_tags({
                    "execution_type": "register_existing",
                    "trained_code_sha256": selection["execution_context"]["code_sha256"],
                })
        elif args.phase in ["all", "train"]:
            selection = train(args.config, retrain=args.retrain)
        else:
            selection = json.loads((ROOT / "models/selection.json").read_text())
        if mlflow is not None:
            mlflow.log_params({"estimator": selection["estimator"], "C": selection["variant"]["C"]})
        if args.phase in ["all", "evaluate"]:
            evaluated = evaluate_model(args.config)
            result = evaluated["baseline"]
            print(f"model.joblib: accuracy={result['accuracy']:.6f}; macro-F1={result['macro_f1']:.6f}")
        if mlflow is not None:
            mlflow.log_metrics(tracking_metrics(
                selection, result if args.phase in ["all", "evaluate", "register"] else None
            ))
            model_info = mlflow.sklearn.log_model(
                sk_model=joblib.load(ROOT / "models/model.joblib"),
                name="model",
                serialization_format="skops",
                skops_trusted_types=[
                    "flight_delay.features.build.FlightFeatures",
                    "numpy.dtype",
                ],
                code_paths=[str(ROOT / "src/flight_delay")],
            )
            model_name = os.environ.get("MLFLOW_MODEL_NAME", "voos-model")
            version = register_model(
                mlflow, model_info.model_uri, model_name, os.environ.get("COMMIT_SHA")
            )
            print(f"MLflow run: {mlflow.active_run().info.run_id}", flush=True)
            print(f"MLflow model: {model_info.model_uri}", flush=True)
            print(f"MLflow registered model: {model_name}, version {version.version}", flush=True)


if __name__ == "__main__":
    main()
