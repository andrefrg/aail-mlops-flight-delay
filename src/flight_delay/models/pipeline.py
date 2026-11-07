"""Classroom estimator, wrapped in the preprocessing required for flight records."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_delay.features.build import FlightFeatures, NUMERIC_COLUMNS, categorical_columns

CLASSES = ["on_time", "moderate_delay", "severe_delay"]


def make_pipeline(variant, config):
    if variant["name"] == "majority":
        return Pipeline(
            [
                ("features", FlightFeatures()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        )
    C = float(variant["C"])
    if C not in [0.1, 1.0, 10.0]:
        raise ValueError("Use the classroom C grid: [0.1, 1.0, 10.0]")
    encoding = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64),
                categorical_columns(),
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
        ],
        sparse_threshold=1.0,
        remainder="drop",
    )
    # Same estimator and explicit parameters as LAIA_PL6/train.py.
    model = LogisticRegression(max_iter=200, C=C)
    return Pipeline(
        [
            ("features", FlightFeatures()),
            ("encode", encoding),
            ("classifier", model),
        ]
    )
