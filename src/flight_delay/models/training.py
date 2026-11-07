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
    if config["model"] != {"estimator": "LogisticRegression", "max_iter": 200} or [
        v["C"] for v in config["variants"]
    ] != [0.1, 1.0, 10.0]:
        raise ValueError("Keep the classroom estimator and C grid")
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
    return {
        "train_rows": len(frame),
        "fit_seconds": round(time.monotonic() - started, 3),
        "convergence_warnings": [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ],
        "iterations": int(np.max(np.atleast_1d(getattr(model[-1], "n_iter_", 0)))),
    }


def before(frame, cutoff, gap_days):
    end = (pd.Timestamp(cutoff) - pd.Timedelta(days=gap_days)).strftime("%Y-%m-%d")
    return frame.loc[frame["FlightDate"].lt(end)].copy(), end


def choose_variant(results, variants):
    table = results.loc[results["model"].ne("majority")]
    means = table.groupby("model")["accuracy"].mean().sort_values(ascending=False, kind="stable")
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
        for variant in variants:
            name = variant["name"]
            result = evaluated[name]
            result_rows.append(
                {
                    "fold": fold["name"],
                    "model": name,
                    "C": variant.get("C"),
                    "train_end_exclusive": train_end,
                    "validation_start": fold["validation_start"],
                    "validation_end_exclusive": fold["validation_end"],
                    "validation_rows": result["rows"],
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "severe_recall": result["classes"]["severe_delay"]["recall"],
                    **information[name],
                }
            )
            print(
                f"Validation {name}: accuracy={result['accuracy']:.6f}; macro-F1={result['macro_f1']:.6f}",
                flush=True,
            )
    results = pd.DataFrame(result_rows)
    variant, scores = choose_variant(results, config["variants"])
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
        "estimator": "LogisticRegression",
        "selection_metric": "mean_validation_accuracy",
        "selection_scores": scores,
        "fit": fit_info,
        "train_end_exclusive": cutoff,
        "provenance": origin,
        "execution_context": execution_context(config_path),
        "model_sha256": sha256(MODELS / "model.joblib"),
    }
    write_json(selection_path, selection)
    print(f"Selected C={variant['C']}; exported models/model.joblib", flush=True)
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
        "estimator": "LogisticRegression",
        "C": selection["variant"]["C"],
        "max_iter": 200,
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
    pd.DataFrame(manifest["monthly_counts"]).to_csv(
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
    reused = (
        "A limpeza de escopo reutilizou os pesos e métricas já verificados, sem retreino. Os bytes do modelo e os módulos de preparação/features/estimador/avaliação foram preservados. A proveniência original e o novo contexto de execução são identificados separadamente."
        if selection.get("scope_cleanup")
        else "Resultados produzidos pela execução de preparação, treino e avaliação desta versão."
    )
    text = f"""# Parte 1 — resultados do baseline

## Modelo e dados

Estimador da aula: LogisticRegression(max_iter=200, C=C), C em [0.1, 1.0, 10.0].
C escolhido pela accuracy média em três folds temporais: {selection["variant"]["C"]}.
Os outros parâmetros do estimador são os valores por omissão da biblioteca.

Preparados {manifest["counts"]["prepared"]:,} voos elegíveis. Durações previstas
inválidas convertidas em ausência: {manifest["counts"]["invalid_duration_to_missing"]}.
Treino com até {config["sample_per_month"]:,} voos por mês; validação e teste usam
todos os voos preparados dos respetivos períodos. Os oito inputs pré-partida
são selecionados explicitamente. A imputação, escala e encoding são aprendidos
só no treino de cada fold. Há {config["gap_days"]} dias de intervalo antes da avaliação.

O modelo final foi ajustado em {selection["fit"]["train_rows"]:,} voos anteriores
a {selection["train_end_exclusive"]} (exclusivo), com {selection["fit"]["iterations"]}
iterações e {len(selection["fit"]["convergence_warnings"])} avisos de convergência.
Não foi reajustado no período de teste.

## Teste: julho a dezembro de 2023

- Voos: {result["rows"]:,}.
- Accuracy: {result["accuracy"]:.6f}; referência maioritária: {evaluated["majority"]["accuracy"]:.6f}.
- Macro-F1: {result["macro_f1"]:.6f}; referência: {evaluated["majority"]["macro_f1"]:.6f}.
- Recall severe_delay: {100 * result["classes"]["severe_delay"]["recall"]:.6f}%.

O modelo prevê quase sempre on_time e praticamente não deteta a classe severa.
A accuracy elevada reflete sobretudo o desbalanceamento. Não se demonstrou
utilidade operacional. O teste já tinha sido observado em experiências anteriores;
C foi selecionado apenas nas janelas de validação. Não se afirma que seja um teste
novo nunca inspecionado. Não existem timestamps de publicação dos rótulos BTS:
a avaliação é cronológica retrospetiva, sem simular esse calendário.

## Entrega aos colegas

- models/model.joblib: pipeline completo, nome base model.
- models/model_metadata.json: esquema, classes, parâmetros, hashes e proveniência.
- reports/tables/: tabelas pequenas para o relatório.
- reports/generated/baseline/: avaliação por classe, matriz de confusão e subgrupos.
- docs/model-handoff.md: interface e comandos propostos para a equipa.

{reused}

Esta parte inclui um Dockerfile de treino, mas não implementa servidor HTTP, Docker de serviço, workflows GitHub Actions,
autenticação, VMs, registry ou publicação. Essas responsabilidades pertencem aos
colegas 2, 3 e 4. Os resultados locais estão prontos para o colega 2 registar no
sistema de tracking da equipa. Os comandos de treino/inferência são uma proposta
a acordar em conjunto; não são apresentados como decisão já aprovada pelo grupo.
"""
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/baseline-results.md").write_text(text, encoding="utf-8")
    (REPORTS / "summary.md").write_text(text, encoding="utf-8")
