"""MLflow presentation of candidate validation; selection remains in training.py."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flight_delay.models.pipeline import make_pipeline

VALIDATION_METRICS = [
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "moderate_recall",
    "severe_recall",
    "severe_precision",
]


def tracking_metrics(reports, selection, result=None):
    table = pd.read_csv(Path(reports) / "cross_validation.csv")
    scores = table.loc[table.model.eq(selection["variant"]["name"])]
    if scores.empty:
        raise ValueError("Missing validation results for selected model")
    metrics = {}
    for field in VALIDATION_METRICS:
        if field not in scores:
            continue  # Historical reports may have fewer metrics.
        values = scores[field].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite validation metric: {field}")
        metrics[f"cv_{field}_mean"] = float(values.mean())
        metrics[f"cv_{field}_std"] = float(values.std())
    if result is not None:
        test = {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "balanced_accuracy": result["balanced_accuracy"],
            "recall_severe_delay": result["classes"]["severe_delay"]["recall"],
            "recall_moderate_delay": result["classes"]["moderate_delay"]["recall"],
            "precision_severe_delay": result["classes"]["severe_delay"]["precision"],
        }
        metrics.update(test)  # Keep existing dashboard keys.
        metrics.update({f"test_{key}": value for key, value in test.items()})
    return metrics


def log_candidate_runs(mlflow, reports, config, selection, execution_type):
    reports = Path(reports)
    table = pd.read_csv(reports / "cross_validation.csv")
    definitions = {v["name"]: v for v in [{"name": "majority"}, *config["variants"]]}
    details_path = reports / "validation_details.json"
    details = json.loads(details_path.read_text()) if details_path.exists() else {}
    for name, scores in table.groupby("model", sort=False):
        if name not in definitions:
            continue
        variant = definitions[name]
        classifier = make_pipeline(variant, config)[-1]
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params(
                {
                    "candidate": name,
                    "estimator": type(classifier).__name__,
                    **{k: v for k, v in classifier.get_params().items() if v is not None},
                }
            )
            mlflow.set_tags(
                {
                    "selected": str(name == selection["variant"]["name"]).lower(),
                    "role": "reference" if name == "majority" else "candidate",
                    "selection_metric": selection["selection_metric"],
                    "execution_type": execution_type,
                    "data_manifest_sha256": selection["provenance"]["data_manifest_sha256"],
                }
            )
            means = {}
            for metric in VALIDATION_METRICS:
                if metric in scores:
                    values = scores[metric].to_numpy(dtype=float)
                    means[f"cv_{metric}_mean"] = float(values.mean())
                    means[f"cv_{metric}_std"] = float(values.std())
            mlflow.log_metrics(means)
            for index, row in enumerate(scores.to_dict("records")):
                mlflow.log_metrics(
                    {f"validation_{m}": float(row[m]) for m in VALIDATION_METRICS if m in row},
                    step=index,
                )
            mlflow.log_dict(
                {
                    "folds": json.loads(scores.to_json(orient="records")),
                    "details": {
                        fold: value["models"][name]
                        for fold, value in details.items()
                        if name in value["models"]
                    },
                    "trained_utc": selection["provenance"]["created_utc"],
                },
                "validation.json",
            )
