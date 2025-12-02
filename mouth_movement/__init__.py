"""
Lightweight mouth movement detection package.

Provides feature extraction from facial landmarks, a simple feedforward
neural network implemented in NumPy, and utilities for training and
real-time inference.
"""

from .features import MouthFeatureExtractor, load_dlib_predictor
from .model import FeedforwardNN
from .training import train_model, evaluate_model
from .inference import MouthMovementDetector

__all__ = [
    "MouthFeatureExtractor",
    "load_dlib_predictor",
    "FeedforwardNN",
    "train_model",
    "evaluate_model",
    "MouthMovementDetector",
]
