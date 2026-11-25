"""Configured classifiers sharing the same train-only flight preprocessing."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from flight_delay.features.build import FlightFeatures, categorical_columns, numeric_columns
from flight_delay.models.catboost import CatBoostClassifier
from flight_delay.models.mlp import TorchMLPClassifier

CLASSES = ["on_time", "moderate_delay", "severe_delay"]


def make_pipeline(variant, config):
    feature_set = variant.get("feature_set", "baseline")
    if feature_set not in ["baseline", "engineered"]:
        raise ValueError(f"Unsupported feature set: {feature_set}")
    engineered = feature_set == "engineered"
    if variant["name"] == "majority":
        return Pipeline(
            [
                ("features", FlightFeatures()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        )
    estimator = variant.get("estimator", "LogisticRegression")
    features = FlightFeatures(include_engineered=engineered)
    categories = categorical_columns(include_engineered=engineered)
    if estimator == "CatBoostClassifier":
        model = CatBoostClassifier(
            cat_features=categories,
            iterations=variant.get("iterations", 250),
            depth=variant.get("depth", 6),
            learning_rate=variant.get("learning_rate", 0.1),
            auto_class_weights=variant.get("auto_class_weights"),
            thread_count=variant.get("thread_count", 2),
            random_seed=config.get("seed", 42),
        )
        return Pipeline([("features", features), ("classifier", model)])
    if estimator == "TorchMLPClassifier":
        model = TorchMLPClassifier(
            hidden_sizes=tuple(variant.get("hidden_sizes", [64, 32])),
            epochs=variant.get("epochs", 10),
            batch_size=variant.get("batch_size", 1024),
            learning_rate=variant.get("learning_rate", 0.001),
            class_weight=variant.get("class_weight"),
            random_state=config.get("seed", 42),
        )
    elif estimator == "LogisticRegression":
        C = float(variant["C"])
        if C not in [0.1, 1.0, 10.0]:
            raise ValueError("Use the classroom C grid: [0.1, 1.0, 10.0]")
        if variant.get("class_weight") not in [None, "balanced"]:
            raise ValueError("Unsupported logistic class_weight")
        parameters = {"max_iter": 200, "C": C}
        if variant.get("class_weight") is not None:
            parameters["class_weight"] = variant["class_weight"]
        model = LogisticRegression(**parameters)
    elif estimator in ["RandomForestClassifier", "ExtraTreesClassifier"]:
        forest_class = (
            RandomForestClassifier if estimator == "RandomForestClassifier" else ExtraTreesClassifier
        )
        model = forest_class(
            n_estimators=variant.get("n_estimators", 60),
            max_depth=variant.get("max_depth", 16),
            min_samples_leaf=variant.get("min_samples_leaf", 30),
            class_weight=variant.get("class_weight", "balanced_subsample"),
            n_jobs=variant.get("n_jobs", 2),
            random_state=config.get("seed", 42),
        )
    elif estimator == "LinearSVC":
        model = LinearSVC(
            C=variant.get("C", 0.1),
            class_weight=variant.get("class_weight", "balanced"),
            dual="auto",
            max_iter=variant.get("max_iter", 5000),
            random_state=config.get("seed", 42),
        )
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")
    encoding = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64),
                categories,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns(include_engineered=engineered),
            ),
        ],
        sparse_threshold=1.0,
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", features),
            ("encode", encoding),
            ("classifier", model),
        ]
    )
