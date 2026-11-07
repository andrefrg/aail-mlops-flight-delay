"""Streaming confusion matrices: evaluation uses every record in the requested period."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from flight_delay.data.audit import BASE_FEATURES
from flight_delay.models.pipeline import CLASSES


def metrics(matrix):
    matrix = np.asarray(matrix)
    true = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(diagonal, true, out=np.zeros(3), where=true != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall != 0
    )
    return {
        "rows": int(matrix.sum()),
        "accuracy": float(diagonal.sum() / matrix.sum()) if matrix.sum() else 0.0,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "classes": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(true[i]),
            }
            for i, label in enumerate(CLASSES)
        },
        "labels": CLASSES,
        "confusion_matrix": matrix.tolist(),
    }


def parquet_files(data, start, end):
    for path in sorted(Path(data).glob("*/month=*/*.parquet")):
        month = path.parent.name.split("=", 1)[1]
        if month >= start[:7] and month <= end[:7]:
            yield path


def evaluate(models, data, start, end, subgroup=False):
    matrices = {name: np.zeros((3, 3), dtype=np.int64) for name in models}
    groups = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    for path in parquet_files(data, start, end):
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["FlightDate"].ge(start) & frame["FlightDate"].lt(end)]
        if frame.empty:
            continue
        truth = frame["target"].astype(str).to_numpy()
        for name, model in models.items():
            predicted = model.predict(frame[BASE_FEATURES])
            matrices[name] += confusion_matrix(truth, predicted, labels=CLASSES)
            if subgroup:
                for field in ["month", "Reporting_Airline", "Origin"]:
                    values = frame[field].astype(str).to_numpy()
                    for group in np.unique(values):
                        mask = values == group
                        groups[(name, field, group)] += confusion_matrix(
                            truth[mask], predicted[mask], labels=CLASSES
                        )
    if not any(matrix.sum() for matrix in matrices.values()):
        raise ValueError(f"No evaluation records in [{start}, {end})")
    return {name: metrics(matrix) for name, matrix in matrices.items()}, groups


def subgroup_table(groups):
    records = []
    for (model, field, group), matrix in sorted(groups.items()):
        result = metrics(matrix)
        records.append(
            {
                "model": model,
                "field": field,
                "group": group,
                "rows": result["rows"],
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "severe_recall": result["classes"]["severe_delay"]["recall"],
                "severe_support": result["classes"]["severe_delay"]["support"],
            }
        )
    return pd.DataFrame(records)
