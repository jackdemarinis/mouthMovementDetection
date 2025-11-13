"""
Training Script for Mouth Movement Detection Neural Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
import sys
from pathlib import Path
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.network import create_model
from data_processing.dataset import load_multiple_sessions, create_data_loaders
from utils.metrics import MetricsCalculator, evaluate_model


class Trainer:
    """
    Trainer for mouth movement detection model.
    """

    def __init__(self, config: dict):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create model
        self.model = create_model(
            model_type="standard",
            input_dim=config['model']['input_size'],
            hidden_dim=config['model']['hidden_size'],
            dropout=config['model']['dropout']
        ).to(self.device)

        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Loss function and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

        # TensorBoard
        if config['logging']['use_tensorboard']:
            log_dir = config['paths']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

    def train_epoch(self, train_loader) -> tuple:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        metrics_calc = MetricsCalculator()

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (features, labels) in enumerate(progress_bar):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            metrics_calc.update(predictions, labels, outputs)

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        metrics = metrics_calc.compute_metrics()

        return avg_loss, metrics

    def validate(self, val_loader) -> tuple:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        metrics_calc = MetricsCalculator()

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # Track metrics
                total_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                metrics_calc.update(predictions, labels, outputs)

        avg_loss = total_loss / len(val_loader)
        metrics = metrics_calc.compute_metrics()

        return avg_loss, metrics

    def train(self, train_loader, val_loader, num_epochs: int):
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            epoch_time = time.time() - start_time

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} (took {epoch_time:.2f}s)")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)
                self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Save best model based on F1 score
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"✓ Saved best model (F1: {val_metrics['f1']:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            patience = self.config['training']['early_stopping_patience']
            if self.epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best F1: {self.best_val_f1:.4f}")
                break

            # Periodic checkpoint
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)

        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train mouth movement detection model")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Override data directory from config")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir

    print("\n" + "="*60)
    print("MOUTH MOVEMENT DETECTION - TRAINING")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Data directory: {config['paths']['data_dir']}")
    print("="*60 + "\n")

    # Load data
    print("Loading data...")
    features, labels = load_multiple_sessions(config['paths']['data_dir'])

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        features, labels,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        balance_classes=config['data']['balance_classes']
    )

    # Save scaler for inference
    import joblib
    scaler_path = os.path.join(config['paths']['checkpoint_dir'], "scaler.pkl")
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Create trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['epochs']
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(trainer.model, test_loader, trainer.device)

    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    if 'roc_auc' in test_metrics:
        print(f"ROC AUC:   {test_metrics['roc_auc']:.4f}")
    print("="*60 + "\n")

    # Check success criteria
    print("Success Criteria Check:")
    print(f"✓ Accuracy ≥ 90%: {'PASS' if test_metrics['accuracy'] >= 0.90 else 'FAIL'} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"✓ F1-Score ≥ 0.85: {'PASS' if test_metrics['f1'] >= 0.85 else 'FAIL'} ({test_metrics['f1']:.4f})")
    print()


if __name__ == "__main__":
    main()
