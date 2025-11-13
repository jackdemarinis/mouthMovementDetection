"""
Neural Network Architecture for Mouth Movement Detection

Implements a lightweight feedforward neural network for binary classification
of mouth states (moving vs. not moving).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MouthMovementNet(nn.Module):
    """
    Lightweight feedforward neural network for mouth movement classification.

    Architecture:
    - Input layer: feature_dim neurons
    - Hidden layer: hidden_dim neurons with ReLU activation
    - Dropout for regularization
    - Output layer: 1 neuron with Sigmoid activation

    The network is designed for real-time CPU inference with minimal overhead.
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 12, dropout: float = 0.2):
        """
        Initialize the neural network.

        Args:
            input_dim: Number of input features (default: 25)
            hidden_dim: Number of hidden layer neurons (default: 12)
            dropout: Dropout probability for regularization (default: 0.2)
        """
        super(MouthMovementNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout

        # Input to hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Hidden to output layer
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using He initialization for ReLU activations.
        """
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Input to hidden with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout during training
        x = self.dropout(x)

        # Hidden to output with sigmoid activation
        x = torch.sigmoid(self.fc2(x))

        return x

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary predictions.

        Args:
            x: Input tensor
            threshold: Classification threshold (default: 0.5)

        Returns:
            Binary predictions (0 or 1)
        """
        with torch.no_grad():
            probabilities = self.forward(x)
            predictions = (probabilities >= threshold).float()
        return predictions

    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MouthMovementNetDeeper(nn.Module):
    """
    Slightly deeper variant with two hidden layers for experimentation.

    This is an optional architecture if the single hidden layer doesn't
    achieve the desired accuracy.
    """

    def __init__(self, input_dim: int = 25, hidden_dim1: int = 16,
                 hidden_dim2: int = 8, dropout: float = 0.2):
        """
        Initialize the deeper network.

        Args:
            input_dim: Number of input features
            hidden_dim1: First hidden layer size
            hidden_dim2: Second hidden layer size
            dropout: Dropout probability
        """
        super(MouthMovementNetDeeper, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He/Xavier initialization."""
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.sigmoid(self.fc3(x))

        return x

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Make binary predictions."""
        with torch.no_grad():
            probabilities = self.forward(x)
            predictions = (probabilities >= threshold).float()
        return predictions


def create_model(model_type: str = "standard", input_dim: int = 25,
                hidden_dim: int = 12, dropout: float = 0.2) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_type: Type of model ("standard" or "deeper")
        input_dim: Input feature dimension
        hidden_dim: Hidden layer size
        dropout: Dropout probability

    Returns:
        Initialized model
    """
    if model_type == "standard":
        model = MouthMovementNet(input_dim, hidden_dim, dropout)
    elif model_type == "deeper":
        model = MouthMovementNetDeeper(input_dim, hidden_dim, hidden_dim // 2, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == "__main__":
    # Test the network
    print("Testing MouthMovementNet...")

    # Create model
    model = MouthMovementNet(input_dim=25, hidden_dim=12, dropout=0.2)

    print(f"Model architecture:\n{model}")
    print(f"\nNumber of parameters: {model.get_num_parameters()}")

    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 25)

    # Forward pass
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test prediction
    predictions = model.predict(dummy_input, threshold=0.5)
    print(f"Predictions: {predictions.squeeze()[:10]}")  # Show first 10

    # Test deeper model
    print("\n" + "="*50)
    print("Testing MouthMovementNetDeeper...")
    deeper_model = MouthMovementNetDeeper()
    print(f"Model architecture:\n{deeper_model}")
    print(f"\nNumber of parameters: {sum(p.numel() for p in deeper_model.parameters())}")

    output_deeper = deeper_model(dummy_input)
    print(f"Output shape: {output_deeper.shape}")
