import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .data import compute_features, load_metadata, normalize_features, save_scaler, split_metadata
from .features import MouthFeatureExtractor, load_dlib_predictor, load_opencv_dnn_face_detector
from .model import FeedforwardNN


def evaluate_model(model: FeedforwardNN, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    probs = model.predict_proba(X)
    preds = (probs >= 0.5).astype(np.float32)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    acc = accuracy_score(y_true, preds)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_model(
    metadata_csv: str,
    predictor_path: Optional[str] = None,
    hidden_dim: int = 12,
    learning_rate: float = 0.05,
    l2_lambda: float = 1e-4,
    epochs: int = 200,
    batch_size: int = 32,
    artifacts_dir: str = "artifacts",
    backend: str = "mediapipe",
    pos_weight: float = 1.0,
) -> Dict[str, float]:
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    df = load_metadata(metadata_csv)
    train_df, val_df, test_df = split_metadata(df)

    dnn_net = load_opencv_dnn_face_detector()
    if backend == "dlib":
        if predictor_path is None:
            raise ValueError("predictor_path is required when backend='dlib'")
        detector, predictor = load_dlib_predictor(predictor_path)
        extractor = MouthFeatureExtractor(detector=detector, predictor=predictor, dnn_net=dnn_net, use_mediapipe=False)
    else:
        extractor = MouthFeatureExtractor(detector=None, predictor=None, dnn_net=dnn_net, use_mediapipe=True)

    extractor.reset_temporal_state()
    X_train, y_train = compute_features(train_df, extractor)
    extractor.reset_temporal_state()
    X_val, y_val = compute_features(val_df, extractor)
    extractor.reset_temporal_state()
    X_test, y_test = compute_features(test_df, extractor)

    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

    model = FeedforwardNN(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
    )
    history = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=10,
        pos_weight=pos_weight,
    )

    metrics = evaluate_model(model, X_test, y_test)

    model_path = os.path.join(artifacts_dir, "mouth_movement_nn.npz")
    scaler_path = os.path.join(artifacts_dir, "feature_scaler.npy")
    history_path = os.path.join(artifacts_dir, "training_history.json")
    model.save(model_path)
    save_scaler(scaler, scaler_path)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "val_accuracy": history.val_accuracy,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    return metrics
