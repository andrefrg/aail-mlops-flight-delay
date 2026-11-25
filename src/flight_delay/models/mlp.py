"""Train a small multilayer perceptron with PyTorch and serve with NumPy."""

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_sizes=(64, 32),
        epochs=10,
        batch_size=1024,
        learning_rate=0.001,
        class_weight=None,
        random_state=42,
    ):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        from threadpoolctl import threadpool_limits
        from torch import nn

        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        labels = np.asarray(y, dtype=str)
        classes, targets = np.unique(labels, return_inverse=True)
        targets = torch.from_numpy(targets.astype(np.int64, copy=False))
        class_weights = compute_class_weight(self.class_weight, classes=classes, y=labels)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        input_sizes = [X.shape[1], *self.hidden_sizes]
        output_sizes = [*self.hidden_sizes, len(classes)]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.random_state)
            linear_layers = [nn.Linear(i, o) for i, o in zip(input_sizes, output_sizes)]
        layers = []
        for layer in linear_layers[:-1]:
            layers.extend([layer, nn.ReLU()])
        model = nn.Sequential(*layers, linear_layers[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        generator = np.random.default_rng(self.random_state)
        X = X.astype(np.float32, copy=False)
        with threadpool_limits(limits=2):
            for _ in range(self.epochs):
                permutation = generator.permutation(len(labels))
                for start in range(0, len(labels), self.batch_size):
                    indices = permutation[start : start + self.batch_size]
                    batch = X[indices]
                    if issparse(batch):
                        batch = batch.toarray()
                    optimizer.zero_grad()
                    loss = loss_fn(model(torch.from_numpy(batch)), targets[indices])
                    loss.backward()
                    optimizer.step()
        self.weights_ = [layer.weight.detach().numpy().T.copy() for layer in linear_layers]
        self.biases_ = [layer.bias.detach().numpy().copy() for layer in linear_layers]
        self.classes_ = classes
        self.n_iter_ = self.epochs
        return self

    def predict(self, X):
        check_is_fitted(self, "weights_")
        X = X.astype(np.float32, copy=False)
        predictions = np.empty(X.shape[0], dtype=np.intp)
        for start in range(0, X.shape[0], self.batch_size):
            stop = start + self.batch_size
            # Keep the one-hot matrix sparse through the first multiplication.
            output = X[start:stop]
            for weights, bias in zip(self.weights_[:-1], self.biases_[:-1]):
                output = np.maximum(output @ weights + bias, 0.0)
            scores = output @ self.weights_[-1] + self.biases_[-1]
            predictions[start:stop] = scores.argmax(axis=1)
        return self.classes_[predictions]
