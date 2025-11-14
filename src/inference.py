"""
Real-Time Speech/Talking Detection Inference

Runs the trained model on webcam feed for real-time speech/talking detection.
Detects when the user is TALKING/SPEAKING, not just general mouth movement.
"""

import torch
import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import time
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent))

from features.facial_landmarks import FacialLandmarkDetector
from features.mouth_features import MouthFeatureExtractor
from models.network import create_model


class SpeechDetectionInference:
    """
    Real-time speech/talking detection using trained model.

    Detects when user is actively TALKING/SPEAKING based on:
    - Mouth movement patterns
    - Speech-specific temporal features
    - Rhythmic lip movements characteristic of speech
    """

    def __init__(self, model_path: str, scaler_path: str = None,
                 device: str = "cpu", threshold: float = 0.5):
        """
        Initialize inference system.

        Args:
            model_path: Path to trained model checkpoint
            scaler_path: Path to feature scaler
            device: Device to run inference on
            threshold: Classification threshold
        """
        self.device = torch.device(device)
        self.threshold = threshold

        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get model configuration
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})

        input_size = model_config.get('input_size', 25)
        hidden_size = model_config.get('hidden_size', 12)
        dropout = model_config.get('dropout', 0.2)

        # Create and load model
        self.model = create_model(
            model_type="standard",
            input_dim=input_size,
            hidden_dim=hidden_size,
            dropout=dropout
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully")
        if 'metrics' in checkpoint:
            print(f"Model metrics: {checkpoint['metrics']}")

        # Load scaler
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pkl")

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print("Warning: Scaler not found. Features will not be normalized.")
            self.scaler = None

        # Initialize feature extractors
        self.landmark_detector = FacialLandmarkDetector(detector_type="mediapipe")
        self.feature_extractor = MouthFeatureExtractor()

        # Performance tracking
        self.fps_history = []
        self.inference_times = []

    def predict(self, features: np.ndarray) -> tuple:
        """
        Make prediction on features.

        Args:
            features: Feature array

        Returns:
            Tuple of (prediction, probability)
        """
        # Normalize features
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            probability = self.model(features_tensor)
            prediction = (probability >= self.threshold).float()

        inference_time = (time.time() - start_time) * 1000  # milliseconds
        self.inference_times.append(inference_time)

        return int(prediction.item()), float(probability.item())

    def run_webcam(self, camera_id: int = 0, show_landmarks: bool = True,
                   show_features: bool = False):
        """
        Run real-time inference on webcam feed.

        Args:
            camera_id: Camera device ID
            show_landmarks: Whether to show facial landmarks
            show_features: Whether to show feature values
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print("\n" + "="*60)
        print("REAL-TIME SPEECH/TALKING DETECTION")
        print("="*60)
        print("Detects when you are TALKING/SPEAKING")
        print("(Not just general mouth movement)")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Reset temporal history")
        print("  'l' - Toggle landmark visualization")
        print("  'f' - Toggle feature display")
        print("="*60 + "\n")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame_start = time.time()

            # Detect mouth landmarks
            mouth_landmarks = self.landmark_detector.get_mouth_landmarks(frame)

            # Create display frame
            display = frame.copy()

            if mouth_landmarks is not None:
                # Get bounding box
                x, y, w, h = self.landmark_detector.get_mouth_bounding_box(mouth_landmarks)

                # Extract features
                features = self.feature_extractor.extract_all_features(
                    frame, mouth_landmarks, (x, y, w, h)
                )

                # Make prediction
                prediction, probability = self.predict(features)

                # Visualize
                if show_landmarks:
                    for lx, ly in mouth_landmarks:
                        cv2.circle(display, (int(lx), int(ly)), 1, (0, 255, 0), -1)

                # Draw bounding box
                box_color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)

                # Display prediction
                label = "TALKING" if prediction == 1 else "NOT TALKING"
                label_color = (0, 255, 0) if prediction == 1 else (0, 0, 255)

                # Background for text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.rectangle(display, (10, 10), (20 + text_size[0], 50 + text_size[1]),
                            (0, 0, 0), -1)

                cv2.putText(display, label, (15, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

                # Display probability
                prob_text = f"Confidence: {probability:.2%}"
                cv2.putText(display, prob_text, (15, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display features if requested
                if show_features:
                    y_offset = 120
                    cv2.putText(display, f"MAR: {features[0]:.3f}", (15, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            else:
                cv2.putText(display, "NO FACE DETECTED", (15, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Calculate and display FPS
            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            self.fps_history.append(current_fps)

            # Keep only last 30 FPS values
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)

            avg_fps = np.mean(self.fps_history)

            # Display FPS
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(display, fps_text, (display.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display inference time
            if len(self.inference_times) > 0:
                avg_inference_time = np.mean(self.inference_times[-30:])
                inference_text = f"Inference: {avg_inference_time:.1f}ms"
                cv2.putText(display, inference_text, (display.shape[1] - 200, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Speech/Talking Detection", display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, display)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord('r'):
                self.feature_extractor.reset_history()
                print("Temporal history reset")
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
            elif key == ord('f'):
                show_features = not show_features
                print(f"Features: {'ON' if show_features else 'OFF'}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}")
        print(f"Average inference time: {np.mean(self.inference_times):.2f}ms")
        print(f"Min inference time: {np.min(self.inference_times):.2f}ms")
        print(f"Max inference time: {np.max(self.inference_times):.2f}ms")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Real-time speech/talking detection")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--scaler", type=str, default=None,
                       help="Path to feature scaler")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Classification threshold (higher = more strict)")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to run inference on")
    parser.add_argument("--no-landmarks", action="store_true",
                       help="Don't show facial landmarks")
    parser.add_argument("--show-features", action="store_true",
                       help="Show feature values")

    args = parser.parse_args()

    # Create inference system
    inference = SpeechDetectionInference(
        model_path=args.model,
        scaler_path=args.scaler,
        device=args.device,
        threshold=args.threshold
    )

    # Run webcam
    inference.run_webcam(
        camera_id=args.camera,
        show_landmarks=not args.no_landmarks,
        show_features=args.show_features
    )


if __name__ == "__main__":
    main()
