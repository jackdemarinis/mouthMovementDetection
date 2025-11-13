"""
Evaluation Metrics for Mouth Movement Detection
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict


class MetricsCalculator:
    """
    Calculates and tracks evaluation metrics for binary classification.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()

    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor,
               probabilities: torch.Tensor = None):
        """
        Update metrics with new batch of predictions.

        Args:
            predictions: Binary predictions (0 or 1)
            labels: Ground truth labels
            probabilities: Prediction probabilities (optional, for ROC)
        """
        # Convert to numpy
        preds_np = predictions.detach().cpu().numpy().flatten()
        labels_np = labels.detach().cpu().numpy().flatten()

        self.all_predictions.extend(preds_np)
        self.all_labels.extend(labels_np)

        if probabilities is not None:
            probs_np = probabilities.detach().cpu().numpy().flatten()
            self.all_probabilities.extend(probs_np)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all classification metrics.

        Returns:
            Dictionary of metrics
        """
        preds = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(labels, preds)

        # Precision, Recall, F1
        metrics['precision'] = precision_score(labels, preds, zero_division=0)
        metrics['recall'] = recall_score(labels, preds, zero_division=0)
        metrics['f1'] = f1_score(labels, preds, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)

            # Specificity (True Negative Rate)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

            # False Positive Rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # ROC AUC (if probabilities are available)
        if len(self.all_probabilities) > 0:
            probs = np.array(self.all_probabilities)
            try:
                metrics['roc_auc'] = roc_auc_score(labels, probs)
            except:
                metrics['roc_auc'] = 0.0

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix array
        """
        preds = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        return confusion_matrix(labels, preds)

    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data.

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if len(self.all_probabilities) == 0:
            raise ValueError("Probabilities not available. Call update() with probabilities.")

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probabilities)

        return roc_curve(labels, probs)

    def plot_confusion_matrix(self, save_path: str = None, normalize: bool = False):
        """
        Plot confusion matrix.

        Args:
            save_path: Path to save the plot (optional)
            normalize: Whether to normalize the matrix
        """
        cm = self.get_confusion_matrix()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', cbar=True)
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0.5, 1.5], ['Not Moving (0)', 'Moving (1)'])
        plt.yticks([0.5, 1.5], ['Not Moving (0)', 'Moving (1)'])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.tight_layout()
        return plt.gcf()

    def plot_roc_curve(self, save_path: str = None):
        """
        Plot ROC curve.

        Args:
            save_path: Path to save the plot (optional)
        """
        if len(self.all_probabilities) == 0:
            print("Warning: Probabilities not available for ROC curve")
            return None

        fpr, tpr, thresholds = self.get_roc_curve()
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.tight_layout()
        return plt.gcf()

    def print_metrics(self, prefix: str = ""):
        """
        Print all metrics in a formatted way.

        Args:
            prefix: Prefix for the output (e.g., "Train", "Val", "Test")
        """
        metrics = self.compute_metrics()

        print(f"\n{'='*60}")
        print(f"{prefix} Metrics" if prefix else "Metrics")
        print(f"{'='*60}")
        print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1']:.4f}")

        if 'specificity' in metrics:
            print(f"Specificity: {metrics['specificity']:.4f}")

        if 'roc_auc' in metrics:
            print(f"ROC AUC:     {metrics['roc_auc']:.4f}")

        if 'true_positives' in metrics:
            print(f"\nConfusion Matrix:")
            print(f"  True Positives:  {metrics['true_positives']}")
            print(f"  True Negatives:  {metrics['true_negatives']}")
            print(f"  False Positives: {metrics['false_positives']}")
            print(f"  False Negatives: {metrics['false_negatives']}")

        print(f"{'='*60}\n")


def evaluate_model(model, data_loader, device, threshold=0.5) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator()

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            probabilities = model(features)
            predictions = (probabilities >= threshold).float()

            # Update metrics
            metrics_calc.update(predictions, labels, probabilities)

    return metrics_calc.compute_metrics()


if __name__ == "__main__":
    # Test metrics calculator
    print("Testing MetricsCalculator...")

    # Simulate some predictions
    np.random.seed(42)
    n_samples = 100

    # Generate fake predictions and labels
    labels = torch.randint(0, 2, (n_samples, 1)).float()
    probabilities = torch.rand(n_samples, 1)
    predictions = (probabilities >= 0.5).float()

    # Create calculator
    calc = MetricsCalculator()
    calc.update(predictions, labels, probabilities)

    # Print metrics
    calc.print_metrics("Test")

    # Plot confusion matrix
    calc.plot_confusion_matrix()
    plt.show()

    # Plot ROC curve
    calc.plot_roc_curve()
    plt.show()
