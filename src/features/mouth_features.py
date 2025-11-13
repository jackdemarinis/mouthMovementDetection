"""
Mouth Feature Extraction Module

Extracts discriminative features from mouth landmarks for movement detection.
"""

import numpy as np
import cv2
from typing import Dict, Optional, List
from scipy.spatial import distance


class MouthFeatureExtractor:
    """
    Extracts features from mouth landmarks for movement classification.

    Features include:
    - Mouth Aspect Ratio (MAR)
    - Inter-landmark distances
    - Temporal derivatives
    - Pixel intensity statistics
    - Edge responses
    """

    def __init__(self, temporal_window: int = 3):
        """
        Initialize the feature extractor.

        Args:
            temporal_window: Number of previous frames to use for temporal features
        """
        self.temporal_window = temporal_window
        self.landmark_history = []
        self.feature_history = []

    def compute_mouth_aspect_ratio(self, mouth_landmarks: np.ndarray) -> float:
        """
        Compute Mouth Aspect Ratio (MAR).

        MAR = (vertical distance) / (horizontal distance)

        Args:
            mouth_landmarks: Array of mouth landmark coordinates (N, 2)

        Returns:
            MAR value
        """
        # For MediaPipe, we use specific landmarks
        # Simplified: use bounding box dimensions
        if len(mouth_landmarks) < 4:
            return 0.0

        x_coords = mouth_landmarks[:, 0]
        y_coords = mouth_landmarks[:, 1]

        # Horizontal distance (mouth width)
        horizontal_dist = np.max(x_coords) - np.min(x_coords)

        # Vertical distance (mouth height)
        vertical_dist = np.max(y_coords) - np.min(y_coords)

        if horizontal_dist == 0:
            return 0.0

        mar = vertical_dist / horizontal_dist

        return mar

    def compute_inter_landmark_distances(self, mouth_landmarks: np.ndarray,
                                        num_samples: int = 10) -> np.ndarray:
        """
        Compute Euclidean distances between pairs of mouth landmarks.

        Args:
            mouth_landmarks: Array of mouth landmarks (N, 2)
            num_samples: Number of distance samples to extract

        Returns:
            Array of distances
        """
        if len(mouth_landmarks) < 2:
            return np.zeros(num_samples)

        distances = []

        # Compute some key distances
        # Top to bottom
        if len(mouth_landmarks) >= 4:
            # Vertical distances
            top_y = np.min(mouth_landmarks[:, 1])
            bottom_y = np.max(mouth_landmarks[:, 1])
            vertical_dist = bottom_y - top_y
            distances.append(vertical_dist)

            # Horizontal distance
            left_x = np.min(mouth_landmarks[:, 0])
            right_x = np.max(mouth_landmarks[:, 0])
            horizontal_dist = right_x - left_x
            distances.append(horizontal_dist)

        # Pairwise distances between evenly spaced landmarks
        num_landmarks = len(mouth_landmarks)
        step = max(1, num_landmarks // (num_samples - len(distances)))

        for i in range(0, num_landmarks, step):
            if len(distances) >= num_samples:
                break
            for j in range(i + step, num_landmarks, step):
                if len(distances) >= num_samples:
                    break
                dist = distance.euclidean(mouth_landmarks[i], mouth_landmarks[j])
                distances.append(dist)

        # Pad if necessary
        while len(distances) < num_samples:
            distances.append(0.0)

        return np.array(distances[:num_samples])

    def compute_temporal_features(self, current_landmarks: np.ndarray) -> np.ndarray:
        """
        Compute temporal features (frame-to-frame changes).

        Args:
            current_landmarks: Current frame's mouth landmarks

        Returns:
            Temporal features (differences from previous frames)
        """
        self.landmark_history.append(current_landmarks)

        # Keep only the required window
        if len(self.landmark_history) > self.temporal_window:
            self.landmark_history.pop(0)

        # If we don't have enough history, return zeros
        if len(self.landmark_history) < 2:
            return np.zeros(10)  # 10 temporal features

        # Compute differences
        temporal_features = []

        # Difference from previous frame
        prev_landmarks = self.landmark_history[-2]
        current_landmarks = self.landmark_history[-1]

        if prev_landmarks.shape == current_landmarks.shape:
            # Mean displacement
            displacement = current_landmarks - prev_landmarks
            mean_disp = np.mean(np.abs(displacement))
            temporal_features.append(mean_disp)

            # Max displacement
            max_disp = np.max(np.abs(displacement))
            temporal_features.append(max_disp)

            # Displacement variance
            disp_var = np.var(displacement)
            temporal_features.append(disp_var)

            # X and Y displacements separately
            mean_x_disp = np.mean(np.abs(displacement[:, 0]))
            mean_y_disp = np.mean(np.abs(displacement[:, 1]))
            temporal_features.append(mean_x_disp)
            temporal_features.append(mean_y_disp)

        # Pad to 10 features
        while len(temporal_features) < 10:
            temporal_features.append(0.0)

        return np.array(temporal_features[:10])

    def compute_intensity_features(self, frame: np.ndarray,
                                   mouth_bbox: tuple) -> np.ndarray:
        """
        Compute pixel intensity statistics within mouth region.

        Args:
            frame: Input frame (BGR)
            mouth_bbox: Bounding box (x, y, width, height)

        Returns:
            Intensity features [mean, variance]
        """
        x, y, w, h = mouth_bbox

        # Ensure valid bbox
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))

        # Extract mouth region
        mouth_region = frame[y:y+h, x:x+w]

        if mouth_region.size == 0:
            return np.array([0.0, 0.0])

        # Convert to grayscale
        if len(mouth_region.shape) == 3:
            gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_mouth = mouth_region

        # Compute statistics
        mean_intensity = np.mean(gray_mouth)
        var_intensity = np.var(gray_mouth)

        return np.array([mean_intensity, var_intensity])

    def compute_edge_features(self, frame: np.ndarray, mouth_bbox: tuple) -> np.ndarray:
        """
        Compute edge response features within mouth region.

        Args:
            frame: Input frame
            mouth_bbox: Bounding box (x, y, width, height)

        Returns:
            Edge features [mean gradient magnitude, max gradient magnitude]
        """
        x, y, w, h = mouth_bbox

        # Ensure valid bbox
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))

        # Extract mouth region
        mouth_region = frame[y:y+h, x:x+w]

        if mouth_region.size == 0:
            return np.array([0.0, 0.0])

        # Convert to grayscale
        if len(mouth_region.shape) == 3:
            gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_mouth = mouth_region

        # Compute gradients
        grad_x = cv2.Sobel(gray_mouth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_mouth, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        mean_grad = np.mean(grad_magnitude)
        max_grad = np.max(grad_magnitude)

        return np.array([mean_grad, max_grad])

    def extract_all_features(self, frame: np.ndarray, mouth_landmarks: np.ndarray,
                            mouth_bbox: tuple) -> np.ndarray:
        """
        Extract all features for the current frame.

        Args:
            frame: Input frame
            mouth_landmarks: Mouth landmark coordinates
            mouth_bbox: Mouth bounding box

        Returns:
            Feature vector of dimension 25:
            - MAR: 1
            - Inter-landmark distances: 10
            - Temporal features: 10
            - Intensity features: 2
            - Edge features: 2
        """
        features = []

        # 1. Mouth Aspect Ratio (1 feature)
        mar = self.compute_mouth_aspect_ratio(mouth_landmarks)
        features.append(mar)

        # 2. Inter-landmark distances (10 features)
        distances = self.compute_inter_landmark_distances(mouth_landmarks, num_samples=10)
        features.extend(distances)

        # 3. Temporal features (10 features)
        temporal = self.compute_temporal_features(mouth_landmarks)
        features.extend(temporal)

        # 4. Intensity statistics (2 features)
        intensity = self.compute_intensity_features(frame, mouth_bbox)
        features.extend(intensity)

        # 5. Edge responses (2 features)
        edges = self.compute_edge_features(frame, mouth_bbox)
        features.extend(edges)

        return np.array(features)

    def reset_history(self):
        """Reset temporal history (e.g., when starting a new video)."""
        self.landmark_history = []
        self.feature_history = []


if __name__ == "__main__":
    # Test feature extraction
    from facial_landmarks import FacialLandmarkDetector

    detector = FacialLandmarkDetector()
    extractor = MouthFeatureExtractor()

    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")
    print("Press 'r' to reset temporal history")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get mouth landmarks
        mouth_landmarks = detector.get_mouth_landmarks(frame)

        if mouth_landmarks is not None:
            # Get bounding box
            x, y, w, h = detector.get_mouth_bounding_box(mouth_landmarks)

            # Extract features
            features = extractor.extract_all_features(frame, mouth_landmarks, (x, y, w, h))

            # Display
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {features[0]:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Features: {len(features)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Feature Extraction", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            extractor.reset_history()
            print("History reset")

    cap.release()
    cv2.destroyAllWindows()
