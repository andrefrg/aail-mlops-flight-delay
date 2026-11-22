"""Compare fixed model candidates on development folds; never evaluate the final test."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flight_delay.data.audit import BASE_FEATURES, sha256  # noqa: E402
from flight_delay.data.prepare import preparation_signature, priority  # noqa: E402
from flight_delay.models.evaluation import evaluate  # noqa: E402
from flight_delay.models.pipeline import make_pipeline  # noqa: E402

def build_model(name, seed=42, variant=None):
    """Keep the classroom alias while using the shared configured model factory."""
    if variant is None:
        config = json.loads((ROOT / "configs/baseline.json").read_text())
        definitions = {item["name"]: item for item in config["variants"]}
        definitions["majority"] = {"name": "majority"}
        definitions["classroom_logistic"] = {"name": "classroom_logistic", "C": 0.1}
        if name not in definitions:
            raise ValueError(f"Unknown candidate: {name}")
        variant = definitions[name]
    return make_pipeline(variant, {"seed": seed})


def development_sample(frame, config, per_month):
    if not 0 < per_month <= config["sample_per_month"]:
        raise ValueError("Sample size must be positive and at most the prepared monthly budget")
    frame = frame.loc[frame.FlightDate.lt(config["holdout_start"])].copy()
    frame["experiment_priority"] = priority(frame.row_id.to_numpy(), config["seed"])
    return (
        frame.sort_values(["month", "experiment_priority", "row_id"])
        .groupby("month", sort=True)
        .head(per_month)
        .drop(columns="experiment_priority")
        .sort_values("row_id")
    )


def validate_protocol(config):
    if config["features"] != BASE_FEATURES or config["gap_days"] < 0:
        raise ValueError("Use the baseline feature allowlist and a nonnegative temporal gap")
    for fold in config["folds"]:
        if not fold["validation_start"] < fold["validation_end"] <= config["holdout_start"]:
            raise ValueError("Validation must end before or at the final holdout start")


def code_hash():
    digest = hashlib.sha256()
    for path in sorted([Path(__file__), *list((ROOT / "src/flight_delay").rglob("*.py"))]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compare(data, output, config, per_month):
    data, output = Path(data), Path(output)
    validate_protocol(config)
    variants = [{"name": "majority"}, *config["variants"]]
    names = [item["name"] for item in variants]
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("Configure unique candidates, excluding the majority reference")
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["preparation_config_sha256"] != preparation_signature(config):
        raise ValueError("Prepared data does not match the preparation configuration")
    sample = development_sample(pd.read_parquet(data / "monthly_sample.parquet"), config, per_month)
    output.mkdir(parents=True, exist_ok=False)
    origin = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "code_sha256": code_hash(),
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "raw_sources": manifest["sources"],
        "versions": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib"]
        },
        "protocol": {
            "selection_metric": "mean_validation_macro_f1",
            "sample_per_month": per_month,
            "validation": "All prepared records in each development validation period",
            "final_holdout_evaluated": False,
            "config": config,
            "note": "Development comparison only; all candidates use the same training rows. No production promotion.",
        },
    }
    (output / "provenance.json").write_text(json.dumps(origin, indent=2) + "\n")
    rows = []
    for fold in config["folds"]:
        cutoff = (
            pd.Timestamp(fold["validation_start"]) - pd.Timedelta(days=config["gap_days"])
        ).strftime("%Y-%m-%d")
        training = sample.loc[sample.FlightDate.lt(cutoff)]
        if training.empty:
            raise ValueError("Empty training fold")
        for variant in variants:
            name = variant["name"]
            model = build_model(name, config["seed"], variant)
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(training[BASE_FEATURES], training.target.astype(str))
            fit_seconds = time.monotonic() - started
            serialized = io.BytesIO()
            joblib.dump(model, serialized, compress=3)
            started = time.monotonic()
            result = evaluate(
                {name: model}, data, fold["validation_start"], fold["validation_end"]
            )[0][name]
            evaluation_seconds = time.monotonic() - started
            rows.append(
                {
                    "fold": fold["name"],
                    "model": name,
                    "train_rows": len(training),
                    "train_end_exclusive": cutoff,
                    "validation_start": fold["validation_start"],
                    "validation_end_exclusive": fold["validation_end"],
                    "validation_rows": result["rows"],
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "moderate_recall": result["classes"]["moderate_delay"]["recall"],
                    "severe_recall": result["classes"]["severe_delay"]["recall"],
                    "severe_precision": result["classes"]["severe_delay"]["precision"],
                    "fit_seconds": fit_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "model_bytes": serialized.tell(),
                    "convergence_warnings": sum(
                        issubclass(w.category, ConvergenceWarning) for w in caught
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(output / "fold_metrics.csv", index=False)
            detail = {
                "metrics": result,
                "estimator": model[-1].get_params(),
                "warnings": [str(w.message) for w in caught],
                "sample_rows_sha256": hashlib.sha256(
                    training.row_id.to_numpy().tobytes()
                ).hexdigest(),
            }
            (output / f"{fold['name']}_{name}.json").write_text(json.dumps(detail, indent=2) + "\n")
            print(
                f"{fold['name']} / {name}: macro-F1={result['macro_f1']:.4f}; fit={fit_seconds:.1f}s",
                flush=True,
            )
            del model, serialized
    summary = (
        pd.DataFrame(rows)
        .groupby("model", sort=False)
        .agg(
            folds=("fold", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            moderate_recall_mean=("moderate_recall", "mean"),
            severe_recall_mean=("severe_recall", "mean"),
            severe_precision_mean=("severe_precision", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
            evaluation_seconds_mean=("evaluation_seconds", "mean"),
            model_bytes_max=("model_bytes", "max"),
            convergence_warnings=("convergence_warnings", "sum"),
        )
        .sort_values("macro_f1_mean", ascending=False, kind="stable")
    )
    summary.to_csv(output / "summary.csv")
    winner = summary.drop(index="majority").index[0]
    (output / "selection.json").write_text(
        json.dumps(
            {
                "screening_winner": winner,
                "metric": "mean_validation_macro_f1",
                "final_test_evaluated": False,
                "promoted": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--sample-per-month", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports/generated/model_experiments"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    compare(args.data, args.output, json.loads(args.config.read_text()), args.sample_per_month)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
