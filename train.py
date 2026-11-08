"""Prepare, train, evaluate and register the flight-delay model in MLflow."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
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
            print("Registering the saved model and recorded test accuracy; no retraining.", flush=True)
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
        if mlflow is not None and args.phase in ["all", "evaluate", "register"]:
            mlflow.log_metric("accuracy", result["accuracy"])
        if mlflow is not None:
            model_info = mlflow.sklearn.log_model(
                sk_model=joblib.load(ROOT / "models/model.joblib"),
                name="model",
                serialization_format="skops",
                skops_trusted_types=[
                    "flight_delay.features.build.FlightFeatures",
                    "numpy.dtype",
                ],
                code_paths=[str(ROOT / "src/flight_delay")],
                registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "voos-model"),
            )
            print(f"MLflow run: {mlflow.active_run().info.run_id}", flush=True)
            print(f"MLflow model: {model_info.model_uri}", flush=True)


if __name__ == "__main__":
    main()
