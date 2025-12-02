from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    val_accuracy: list


class FeedforwardNN:
    """
    Single-hidden-layer feedforward neural network implemented in NumPy.
    Designed for fast CPU inference on small feature vectors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 12,
        learning_rate: float = 0.05,
        l2_lambda: float = 1e-4,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # He initialization for the hidden layer
        self.W1 = rng.standard_normal((input_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_dim, 1)).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, 1), dtype=np.float32)
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2, cache

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, pos_weight: float = 1.0) -> float:
        # Weighted binary cross-entropy with L2 regularization
        eps = 1e-7
        m = y_true.shape[0]
        weights = pos_weight * y_true + (1 - y_true)
        bce = -(weights * (y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))).mean()
        reg = (self.l2_lambda / (2 * m)) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(bce + reg)

    def backward(self, cache: Dict[str, np.ndarray], y_true: np.ndarray) -> None:
        X, z1, a1, z2, a2 = (
            cache["X"],
            cache["z1"],
            cache["a1"],
            cache["z2"],
            cache["a2"],
        )
        m = y_true.shape[0]

        dz2 = a2 - y_true
        dW2 = (a1.T @ dz2) / m + (self.l2_lambda / m) * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (dz2 @ self.W2.T) * sigmoid_derivative(z1)
        dW1 = (X.T @ dz1) / m + (self.l2_lambda / m) * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X)
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 10,
        pos_weight: float = 1.0,
    ) -> TrainingHistory:
        history = TrainingHistory(train_loss=[], val_loss=[], val_accuracy=[])
        best_val = float("inf")
        wait = 0

        for epoch in range(epochs):
            # Shuffle training data each epoch
            perm = np.random.permutation(X_train.shape[0])
            X_train = X_train[perm]
            y_train = y_train[perm]

            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                xb = X_train[start:end]
                yb = y_train[start:end]
                probs, cache = self.forward(xb)
                self.backward(cache, yb)

            train_probs, _ = self.forward(X_train)
            val_probs, _ = self.forward(X_val)
            train_loss = self.compute_loss(train_probs, y_train, pos_weight)
            val_loss = self.compute_loss(val_probs, y_val, pos_weight)
            val_pred = (val_probs >= 0.5).astype(np.float32)
            val_acc = float(np.mean(val_pred == y_val))

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.val_accuracy.append(val_acc)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return history

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            learning_rate=self.learning_rate,
            l2_lambda=self.l2_lambda,
        )

    @classmethod
    def load(cls, path: str) -> "FeedforwardNN":
        data = np.load(path, allow_pickle=True)
        model = cls(
            input_dim=data["W1"].shape[0],
            hidden_dim=data["W1"].shape[1],
            learning_rate=float(data["learning_rate"]),
            l2_lambda=float(data["l2_lambda"]),
        )
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        return model
