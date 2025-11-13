"""
PyTorch Dataset for Mouth Movement Detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json


class MouthMovementDataset(Dataset):
    """
    PyTorch Dataset for mouth movement classification.

    Loads features and labels from processed data directories.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize the dataset.

        Args:
            features: Feature array of shape (N, feature_dim)
            labels: Label array of shape (N,)
            transform: Optional transform to apply to features
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)  # Shape: (N, 1)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, label)
        """
        features = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            features = self.transform(features)

        return features, label


def load_session_data(session_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from a session directory.

    Args:
        session_dir: Path to session directory

    Returns:
        Tuple of (features, labels)
    """
    features_path = os.path.join(session_dir, "features.npy")
    labels_path = os.path.join(session_dir, "labels.npy")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found: {features_path}")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    features = np.load(features_path)
    labels = np.load(labels_path)

    # Filter out unlabeled samples (-1)
    valid_indices = labels != -1
    features = features[valid_indices]
    labels = labels[valid_indices]

    print(f"Loaded session: {session_dir}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Moving: {np.sum(labels == 1)} ({np.mean(labels) * 100:.1f}%)")
    print(f"  Not moving: {np.sum(labels == 0)} ({(1 - np.mean(labels)) * 100:.1f}%)")

    return features, labels


def load_multiple_sessions(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine data from multiple sessions.

    Args:
        data_dir: Directory containing session subdirectories

    Returns:
        Combined features and labels
    """
    all_features = []
    all_labels = []

    # Find all session directories
    session_dirs = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    print(f"\nFound {len(session_dirs)} sessions in {data_dir}")

    for session_dir in session_dirs:
        session_path = os.path.join(data_dir, session_dir)

        try:
            features, labels = load_session_data(session_path)
            all_features.append(features)
            all_labels.append(labels)
        except FileNotFoundError as e:
            print(f"Skipping {session_dir}: {e}")
            continue

    if len(all_features) == 0:
        raise ValueError(f"No valid sessions found in {data_dir}")

    # Combine all data
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)

    print(f"\nCombined dataset:")
    print(f"  Total samples: {len(combined_labels)}")
    print(f"  Feature dimension: {combined_features.shape[1]}")
    print(f"  Moving: {np.sum(combined_labels == 1)} ({np.mean(combined_labels) * 100:.1f}%)")
    print(f"  Not moving: {np.sum(combined_labels == 0)}")

    return combined_features, combined_labels


def create_balanced_dataset(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance the dataset by sampling equal numbers from each class.

    Args:
        features: Feature array
        labels: Label array

    Returns:
        Balanced features and labels
    """
    moving_indices = np.where(labels == 1)[0]
    not_moving_indices = np.where(labels == 0)[0]

    # Sample the same number from each class
    min_samples = min(len(moving_indices), len(not_moving_indices))

    balanced_moving = np.random.choice(moving_indices, min_samples, replace=False)
    balanced_not_moving = np.random.choice(not_moving_indices, min_samples, replace=False)

    balanced_indices = np.concatenate([balanced_moving, balanced_not_moving])
    np.random.shuffle(balanced_indices)

    balanced_features = features[balanced_indices]
    balanced_labels = labels[balanced_indices]

    print(f"\nBalanced dataset:")
    print(f"  Total samples: {len(balanced_labels)}")
    print(f"  Moving: {np.sum(balanced_labels == 1)} (50.0%)")
    print(f"  Not moving: {np.sum(balanced_labels == 0)} (50.0%)")

    return balanced_features, balanced_labels


def create_data_loaders(features: np.ndarray, labels: np.ndarray,
                       batch_size: int = 32,
                       train_split: float = 0.7,
                       val_split: float = 0.15,
                       test_split: float = 0.15,
                       balance_classes: bool = True,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Create train, validation, and test data loaders.

    Args:
        features: Feature array
        labels: Label array
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        balance_classes: Whether to balance classes
        random_state: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Balance classes if requested
    if balance_classes:
        features, labels = create_balanced_dataset(features, labels)

    # Split data
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=test_split,
        random_state=random_state,
        stratify=labels
    )

    # Second split: separate train and validation
    val_size_adjusted = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train) / len(features) * 100:.1f}%)")
    print(f"  Val: {len(X_val)} samples ({len(X_val) / len(features) * 100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test) / len(features) * 100:.1f}%)")

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create datasets
    train_dataset = MouthMovementDataset(X_train, y_train)
    val_dataset = MouthMovementDataset(X_val, y_val)
    test_dataset = MouthMovementDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    # Test data loading
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw",
                       help="Directory containing session data")
    args = parser.parse_args()

    # Load data
    features, labels = load_multiple_sessions(args.data_dir)

    # Create data loaders
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        features, labels,
        batch_size=32,
        balance_classes=True
    )

    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Test a batch
    for batch_features, batch_labels in train_loader:
        print(f"\nBatch example:")
        print(f"  Features shape: {batch_features.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Features range: [{batch_features.min():.3f}, {batch_features.max():.3f}]")
        print(f"  Labels: {batch_labels.squeeze()[:10]}")
        break
