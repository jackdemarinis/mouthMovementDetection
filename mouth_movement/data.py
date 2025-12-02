import os
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .features import MouthFeatureExtractor


def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Expected CSV format:
        filepath,label
        path/to/frame.jpg,1
        path/to/frame2.jpg,0
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at '{csv_path}'. Provide a valid --metadata path.")
    df = pd.read_csv(csv_path)
    if not {"filepath", "label"}.issubset(df.columns):
        raise ValueError("metadata CSV must have 'filepath' and 'label' columns")
    return df


def split_metadata(
    df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, stratify=df["label"], random_state=seed)
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative_val_size, stratify=temp_df["label"], random_state=seed
    )
    return train_df, val_df, test_df


def compute_features(
    df: pd.DataFrame, extractor: MouthFeatureExtractor, drop_missing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[int] = []
    feature_len: int = 0
    missing_files = 0
    unreadable = 0
    unsupported = 0
    no_features = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        path = row["filepath"]
        if not os.path.exists(path):
            missing_files += 1
            continue
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            unreadable += 1
            continue
        # Normalize frame type to 8-bit BGR for dlib/OpenCV
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            pass
        else:
            # Skip unsupported shapes
            unsupported += 1
            continue
        try:
            feature_vector, detection = extractor.extract_features(frame)
        except Exception:
            no_features += 1
            continue
        if feature_vector is None:
            if drop_missing:
                no_features += 1
                continue
            if feature_len == 0:
                # default to 38 features if nothing has been computed yet
                feature_len = 38
            feature_vector = np.zeros((feature_len,), dtype=np.float32)  # fallback placeholder
        else:
            feature_len = len(feature_vector)
        feats.append(feature_vector)
        labels.append(int(row["label"]))

    if len(feats) == 0:
        raise ValueError(
            "No usable samples extracted. "
            f"Missing files: {missing_files}, unreadable: {unreadable}, unsupported format: {unsupported}, "
            f"no landmarks/features: {no_features}. "
            "Ensure metadata paths are correct and frames contain a visible face."
        )

    return np.vstack(feats), np.array(labels, dtype=np.float32).reshape(-1, 1)


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_val_norm, X_test_norm, scaler


def save_scaler(scaler: StandardScaler, path: str) -> None:
    np.save(path, {"mean": scaler.mean_, "scale": scaler.scale_})


def load_scaler(path: str) -> StandardScaler:
    arr = np.load(path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = arr["mean"]
    scaler.scale_ = arr["scale"]
    scaler.n_features_in_ = scaler.mean_.shape[0]
    return scaler
