"""Prepare, train, evaluate and register the flight-delay model in MLflow."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import sys

import joblib
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from flight_delay.data.prepare import prepare
from flight_delay.data.audit import sha256
from flight_delay.models.training import train, evaluate_model
from flight_delay.models.tracking import log_candidate_runs, log_selected_model
from flight_delay.models.tracking import tracking_metrics as validation_metrics


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
    return validation_metrics(ROOT / "reports/generated/baseline", selection, result)


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

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://10.17.0.234:8080"))
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "voos-classification"))

    config = json.loads(args.config.read_text())
    reused = (ROOT / "models/selection.json").exists() and not args.retrain
    execution_type = (
        "register_existing"
        if args.phase == "register"
        else "evaluate_existing"
        if args.phase == "evaluate"
        else "reuse_existing"
        if reused
        else "train"
    )
    with (
        mlflow.start_run(run_name=f"selection-{execution_type}")
        if mlflow is not None
        else nullcontext()
    ):
        if mlflow is not None:
            mlflow.log_params(
                {
                    "selection_metric": config.get("selection_metric", "mean_validation_accuracy"),
                    "candidate_count": len(config["variants"]),
                    "gap_days": config["gap_days"],
                    "sample_per_month": config["sample_per_month"],
                }
            )
            mlflow.set_tags({"role": "selection", "execution_type": execution_type})
        if args.phase == "register":
            selection, result = recorded_evaluation(args.config)
            print("Registering the saved model and recorded evaluation; no retraining.", flush=True)
            if mlflow is not None:
                mlflow.set_tags(
                    {
                        "execution_type": "register_existing",
                        "trained_code_sha256": selection["execution_context"]["code_sha256"],
                    }
                )
        elif args.phase in ["all", "train"]:
            selection = train(args.config, retrain=args.retrain)
        else:
            selection = json.loads((ROOT / "models/selection.json").read_text())
        if mlflow is not None:
            mlflow.log_params(
                {
                    "estimator": selection["estimator"],
                    "selected_variant": selection["variant"]["name"],
                    "selection_metric": selection["selection_metric"],
                    **{
                        key: value
                        for key, value in selection["variant"].items()
                        if key not in ["name", "estimator"] and value is not None
                    },
                }
            )
            mlflow.set_tags(
                {
                    "execution_type": execution_type,
                    "role": "selection",
                    "data_manifest_sha256": selection["provenance"]["data_manifest_sha256"],
                    "config_sha256": selection["provenance"]["config_sha256"],
                    "trained_git_commit": selection["provenance"]["git_commit"],
                    "trained_git_dirty": str(selection["provenance"]["git_dirty"]).lower(),
                }
            )
        if args.phase in ["all", "evaluate"]:
            evaluated = evaluate_model(args.config)
            result = evaluated["baseline"]
            print(
                f"model.joblib: accuracy={result['accuracy']:.6f}; macro-F1={result['macro_f1']:.6f}"
            )
        if mlflow is not None:
            mlflow.log_metrics(
                tracking_metrics(
                    selection, result if args.phase in ["all", "evaluate", "register"] else None
                )
            )
            reports = ROOT / "reports/generated/baseline"
            log_candidate_runs(mlflow, reports, config, selection, execution_type)
            for path in [
                args.config,
                ROOT / "models/selection.json",
                ROOT / "models/model_metadata.json",
                ROOT / "data/processed/manifest.json",
                reports / "cross_validation.csv",
                reports / "validation_details.json",
                reports / "holdout.json",
                reports / "holdout_subgroups.csv",
            ]:
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path="selection")
            model_info = log_selected_model(
                mlflow, joblib.load(ROOT / "models/model.joblib"), ROOT / "src/flight_delay"
            )
            model_name = os.environ.get("MLFLOW_MODEL_NAME", "voos-model")
            version = register_model(
                mlflow, model_info.model_uri, model_name, os.environ.get("COMMIT_SHA")
            )
            client = mlflow.MlflowClient()
            for key, value in {
                "estimator": selection["estimator"],
                "selected_variant": selection["variant"]["name"],
                "selection_metric": selection["selection_metric"],
                "data_manifest_sha256": selection["provenance"]["data_manifest_sha256"],
            }.items():
                client.set_model_version_tag(model_name, version.version, key, value)
            print(
                f"Selected: {selection['variant']['name']} ({selection['selection_metric']})",
                flush=True,
            )
            print(f"MLflow run: {mlflow.active_run().info.run_id}", flush=True)
            print(f"MLflow model: {model_info.model_uri}", flush=True)
            print(f"MLflow registered model: {model_name}, version {version.version}", flush=True)


if __name__ == "__main__":
    main()
