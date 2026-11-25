"""Keep the native CatBoost model in a portable sklearn pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class CatBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        cat_features,
        iterations=250,
        depth=6,
        learning_rate=0.1,
        auto_class_weights=None,
        thread_count=2,
        random_seed=42,
    ):
        self.cat_features = cat_features
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.auto_class_weights = auto_class_weights
        self.thread_count = thread_count
        self.random_seed = random_seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier as NativeCatBoost

        model = NativeCatBoost(
            **self.get_params(),
            loss_function="MultiClass",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X, y)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.cbm"
            model.save_model(str(path))
            self.model_bytes_ = np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()
        self.classes_ = model.classes_
        return self

    def predict(self, X):
        from catboost import CatBoostClassifier as NativeCatBoost

        check_is_fitted(self, "model_bytes_")
        model = NativeCatBoost().load_model(blob=self.model_bytes_.tobytes())
        return np.asarray(model.predict(X, thread_count=self.thread_count)).ravel()
