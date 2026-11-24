"""Part 1: temporal model selection, evaluation and a portable joblib export."""

from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import subprocess
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from flight_delay.data.audit import BASE_FEATURES, sha256
from flight_delay.data.prepare import ROOT, preparation_signature, write_json
from flight_delay.models.evaluation import evaluate, subgroup_table
from flight_delay.models.pipeline import CLASSES, make_pipeline

DATA = ROOT / "data/processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports/generated/baseline"


def code_fingerprint():
    digest = hashlib.sha256()
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    files += [ROOT / "train.py"]
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(config_path):
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, OSError):
            return "unavailable"

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain") != "",
        "code_sha256": code_fingerprint(),
        "config_sha256": sha256(config_path),
        "data_manifest_sha256": sha256(DATA / "manifest.json"),
        "versions": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib"]
        },
    }


def load_config(config_path):
    config = json.loads(Path(config_path).read_text())
    if config["features"] != BASE_FEATURES:
        raise ValueError("Feature configuration does not match the implemented allowlist")
    if config.get("selection_metric", "mean_validation_accuracy") not in [
        "mean_validation_accuracy",
        "mean_validation_macro_f1",
    ]:
        raise ValueError("Unsupported selection metric")
    names = [variant["name"] for variant in config["variants"]]
    if not names or len(names) != len(set(names)) or "majority" in names:
        raise ValueError("Candidate names must be unique and exclude the majority reference")
    for variant in config["variants"]:
        make_pipeline(variant, config)
    manifest = json.loads((DATA / "manifest.json").read_text())
    expected = (
        preparation_signature(config)
        if "preparation_config_sha256" in manifest
        else sha256(config_path)
    )
    actual = manifest.get("preparation_config_sha256", manifest["config_sha256"])
    if actual != expected:
        raise ValueError("Preparation settings changed; rebuild the prepared data")
    for fold in config["folds"]:
        if not fold["validation_start"] < fold["validation_end"] <= config["holdout_start"]:
            raise ValueError("Invalid temporal fold or overlap with the test period")
    return config


def fit(model, frame):
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(frame[BASE_FEATURES], frame["target"].astype(str))
    iterations = getattr(model[-1], "n_iter_", None)
    return {
        "train_rows": len(frame),
        "fit_seconds": round(time.monotonic() - started, 3),
        "convergence_warnings": [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ],
        "iterations": int(np.max(np.atleast_1d(iterations))) if iterations is not None else None,
        "trees": getattr(model[-1], "n_estimators", None),
    }


def before(frame, cutoff, gap_days):
    end = (pd.Timestamp(cutoff) - pd.Timedelta(days=gap_days)).strftime("%Y-%m-%d")
    return frame.loc[frame["FlightDate"].lt(end)].copy(), end


def choose_variant(results, variants, metric="accuracy", expected_folds=None):
    """Select by mean development score; exact ties follow configuration order."""
    if metric not in ["accuracy", "macro_f1"]:
        raise ValueError("Unsupported selection metric")
    names = [item["name"] for item in variants]
    table = results.loc[results["model"].isin(names)]
    if set(table["model"]) != set(names) or not np.isfinite(table[metric]).all():
        raise ValueError("Missing or nonfinite candidate validation scores")
    if expected_folds is not None:
        counts = table.groupby("model")["fold"].nunique()
        if not counts.eq(expected_folds).all() or table.duplicated(["model", "fold"]).any():
            raise ValueError("Every candidate must have one result for every fold")
    means = table.groupby("model")[metric].mean().reindex(names)
    means = means.sort_values(ascending=False, kind="stable")
    selected = next(item for item in variants if item["name"] == means.index[0])
    return selected, means.to_dict()


def execution_context(config_path):
    return {
        "code_sha256": code_fingerprint(),
        "config_sha256": sha256(config_path),
        "data_manifest_sha256": sha256(DATA / "manifest.json"),
    }


def train(config_path, retrain=False):
    config = load_config(config_path)
    selection_path = MODELS / "selection.json"
    if selection_path.exists() and not retrain:
        previous = json.loads(selection_path.read_text())
        if (
            previous.get("execution_context") == execution_context(config_path)
            and (MODELS / "model.joblib").exists()
            and previous["model_sha256"] == sha256(MODELS / "model.joblib")
        ):
            print(
                "Using the existing verified model; --retrain explicitly repeats training.",
                flush=True,
            )
            return previous
        raise ValueError(
            "Code/configuration/data changed. Review the protocol and use --retrain to create a new recorded run."
        )
    if retrain:
        (REPORTS / "holdout.json").unlink(missing_ok=True)
        (REPORTS / "holdout_subgroups.csv").unlink(missing_ok=True)
    sample = pd.read_parquet(DATA / "monthly_sample.parquet")
    sample = sample.loc[sample["FlightDate"].lt(config["holdout_start"])].copy()
    variants = [{"name": "majority"}] + config["variants"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    result_rows = []
    validation_details = {}
    origin = provenance(config_path)
    for fold in config["folds"]:
        training, train_end = before(sample, fold["validation_start"], config["gap_days"])
        if training.empty:
            raise ValueError("Empty training fold")
        models, information = {}, {}
        for variant in variants:
            name = variant["name"]
            models[name] = make_pipeline(variant, config)
            information[name] = fit(models[name], training)
            print(f"Fit {fold['name']} / {name}: {len(training):,} sampled rows", flush=True)
        evaluated, _ = evaluate(models, DATA, fold["validation_start"], fold["validation_end"])
        validation_details[fold["name"]] = {
            "train_end_exclusive": train_end,
            "train_rows_sha256": hashlib.sha256(training.row_id.to_numpy().tobytes()).hexdigest(),
            "models": evaluated,
        }
        for variant in variants:
            name = variant["name"]
            result = evaluated[name]
            result_rows.append(
                {
                    "fold": fold["name"],
                    "model": name,
                    "feature_set": variant.get("feature_set", "baseline"),
                    "C": variant.get("C"),
                    "train_end_exclusive": train_end,
                    "validation_start": fold["validation_start"],
                    "validation_end_exclusive": fold["validation_end"],
                    "validation_rows": result["rows"],
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "moderate_recall": result["classes"]["moderate_delay"]["recall"],
                    "severe_precision": result["classes"]["severe_delay"]["precision"],
                    "severe_recall": result["classes"]["severe_delay"]["recall"],
                    **information[name],
                }
            )
            print(
                f"Validation {name}: accuracy={result['accuracy']:.6f}; macro-F1={result['macro_f1']:.6f}",
                flush=True,
            )
    results = pd.DataFrame(result_rows)
    selection_metric = config.get("selection_metric", "mean_validation_accuracy")
    metric = selection_metric.removeprefix("mean_validation_")
    variant, scores = choose_variant(results, config["variants"], metric, len(config["folds"]))
    write_json(REPORTS / "validation_details.json", validation_details)
    training, cutoff = before(sample, config["holdout_start"], config["gap_days"])
    model = make_pipeline(variant, config)
    fit_info = fit(model, training)
    majority = make_pipeline({"name": "majority"}, config)
    fit(majority, training)
    joblib.dump(model, MODELS / "model.joblib", compress=3)
    joblib.dump(majority, MODELS / "majority.joblib", compress=3)
    results.to_csv(REPORTS / "cross_validation.csv", index=False)
    selection = {
        "variant": variant,
        "estimator": type(model[-1]).__name__,
        "feature_set": variant.get("feature_set", "baseline"),
        "selection_metric": selection_metric,
        "selection_scores": scores,
        "fit": fit_info,
        "train_end_exclusive": cutoff,
        "provenance": origin,
        "execution_context": execution_context(config_path),
        "model_sha256": sha256(MODELS / "model.joblib"),
    }
    write_json(selection_path, selection)
    print(
        f"Selected {variant['name']} by {selection_metric}; exported models/model.joblib",
        flush=True,
    )
    return selection


def evaluate_model(config_path):
    config = load_config(config_path)
    selection = json.loads((MODELS / "selection.json").read_text())
    if selection["execution_context"] != execution_context(config_path):
        raise ValueError("Execution context changed after selection")
    if selection["model_sha256"] != sha256(MODELS / "model.joblib"):
        raise ValueError("Model bytes changed after selection")
    report_path = REPORTS / "holdout.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["model_sha256"] != selection["model_sha256"]:
            raise ValueError("Test report belongs to another model")
        evaluated = report["models"]
        print("Using the recorded evaluation of this exact model.", flush=True)
    else:
        models = {
            "baseline": joblib.load(MODELS / "model.joblib"),
            "majority": joblib.load(MODELS / "majority.joblib"),
        }
        evaluated, groups = evaluate(
            models, DATA, config["holdout_start"], config["end_exclusive"], subgroup=True
        )
        subgroup_table(groups).to_csv(REPORTS / "holdout_subgroups.csv", index=False)
        write_json(
            report_path,
            {
                "period_start": config["holdout_start"],
                "period_end_exclusive": config["end_exclusive"],
                "model_sha256": selection["model_sha256"],
                "models": evaluated,
                "note": "This period was already inspected in earlier project experiments; it is not a new unseen test.",
            },
        )
    metadata = {
        "name": "model",
        "file": "model.joblib",
        "format": "joblib",
        "estimator": selection["estimator"],
        "feature_set": selection["variant"].get("feature_set", "baseline"),
        "variant": selection["variant"],
        "selection_metric": selection["selection_metric"],
        "C": selection["variant"].get("C"),
        "max_iter": joblib.load(MODELS / "model.joblib")[-1].get_params().get("max_iter"),
        "fit": selection["fit"],
        "trained_before_exclusive": selection["train_end_exclusive"],
        "raw_features": BASE_FEATURES,
        "classes": CLASSES,
        "sha256": selection["model_sha256"],
        "provenance": selection["provenance"],
        "scope_cleanup": selection.get("scope_cleanup"),
        "purpose": "Complete preprocessing + evaluated classifier, handed off to the API/infrastructure owners. No post-test refit.",
    }
    write_json(MODELS / "model_metadata.json", metadata)
    write_results(config, selection, evaluated)
    return evaluated


def write_results(config, selection, evaluated):
    tables = ROOT / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": name,
                "period": "2023-07_to_2023-12",
                "rows": result["rows"],
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "severe_recall": result["classes"]["severe_delay"]["recall"],
            }
            for name, result in evaluated.items()
        ]
    ).to_csv(tables / "baseline_test_metrics.csv", index=False)
    pd.read_csv(REPORTS / "cross_validation.csv").drop(columns="convergence_warnings").to_csv(
        tables / "baseline_cross_validation.csv", index=False
    )
    manifest = json.loads((DATA / "manifest.json").read_text())
    pd.DataFrame(manifest["monthly_counts"])[["month", "rows"]].to_csv(
        tables / "prepared_monthly_counts.csv", index=False
    )
    cv = pd.read_csv(REPORTS / "cross_validation.csv")
    splits = []
    for fold in config["folds"]:
        cutoff = (
            pd.Timestamp(fold["validation_start"]) - pd.Timedelta(days=config["gap_days"])
        ).strftime("%Y-%m-%d")
        row = cv.loc[cv["fold"].eq(fold["name"])].iloc[0]
        splits.append(
            {
                "fold": fold["name"],
                "train_start": "2022-01-01",
                "train_end_exclusive": cutoff,
                "evaluation_start": fold["validation_start"],
                "evaluation_end_exclusive": fold["validation_end"],
                "sample_train_rows": int(row["train_rows"]),
                "evaluation_rows": int(row["validation_rows"]),
            }
        )
    splits.append(
        {
            "fold": "final_test",
            "train_start": "2022-01-01",
            "train_end_exclusive": selection["train_end_exclusive"],
            "evaluation_start": config["holdout_start"],
            "evaluation_end_exclusive": config["end_exclusive"],
            "sample_train_rows": selection["fit"]["train_rows"],
            "evaluation_rows": evaluated["baseline"]["rows"],
        }
    )
    pd.DataFrame(splits).to_csv(tables / "temporal_splits.csv", index=False)
    result = evaluated["baseline"]
    text = f"""# Seleção automática — resultados

Candidato selecionado: {selection["variant"]["name"]} ({selection["estimator"]}).
Critério: {selection["selection_metric"]}; média aritmética dos folds, empate pela
ordem da configuração. A referência maioritária não é candidata à promoção.

Preparados {manifest["counts"]["prepared"]:,} voos elegíveis; treino com até
{config["sample_per_month"]:,} por mês. Seleção pelos folds de desenvolvimento;
o teste final não escolhe o modelo. Pré-processamento ajustado só no treino.

O vencedor foi ajustado em {selection["fit"]["train_rows"]:,} exemplos anteriores
a {selection["train_end_exclusive"]} (exclusivo), sem refit posterior no teste.

## Avaliação no teste histórico

- Período: {config["holdout_start"]} até {config["end_exclusive"]} (exclusivo).
- Voos: {result["rows"]:,}.
- Accuracy: {result["accuracy"]:.6f}; maioria: {evaluated["majority"]["accuracy"]:.6f}.
- Macro-F1: {result["macro_f1"]:.6f}; maioria: {evaluated["majority"]["macro_f1"]:.6f}.
- Recall severo: {100 * result["classes"]["severe_delay"]["recall"]:.4f}%.
- Precision severa: {100 * result["classes"]["severe_delay"]["precision"]:.4f}%.

O teste já foi observado em experiências anteriores: não é um novo teste nunca
visto. Não existem timestamps de publicação dos rótulos BTS; a avaliação é
cronológica retrospetiva. Maior macro-F1 não garante utilidade operacional.

O pipeline completo está em models/model.joblib. A configuração e os resultados
por fold permitem auditar a escolha. A publicação no MLflow identifica os
candidatos, o critério e a versão vencedora; staging/production seguem a pipeline
de testes existente. Não se compara automaticamente com execuções históricas
que usem outros dados ou protocolos de validação.
"""
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/baseline-results.md").write_text(text, encoding="utf-8")
    (REPORTS / "summary.md").write_text(text, encoding="utf-8")
