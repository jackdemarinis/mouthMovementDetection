"""
Mouth Feature Extraction Module

Extracts discriminative features from mouth landmarks for speech/talking detection.
Optimized to distinguish speech from other mouth movements (chewing, yawning, etc.)
"""

import numpy as np
import cv2
from typing import Dict, Optional, List
from scipy.spatial import distance


class MouthFeatureExtractor:
    """
    Extracts features from mouth landmarks for speech/talking detection.

    Features include:
    - Mouth Aspect Ratio (MAR)
    - Inter-landmark distances
    - Speech-specific temporal features (periodicity, rhythm, consistency)
    - Velocity and acceleration patterns
    - Pixel intensity statistics
    - Edge responses
    """

    def __init__(self, temporal_window: int = 15):
        """
        Initialize the feature extractor.

        Args:
            temporal_window: Number of previous frames to use for temporal features
                           Default: 15 frames (~0.5s at 30fps) for speech pattern detection
        """
        self.temporal_window = temporal_window
        self.landmark_history = []
        self.feature_history = []
        self.mar_history = []  # Track MAR over time for periodicity analysis

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

    def compute_temporal_features(self, current_landmarks: np.ndarray, current_mar: float) -> np.ndarray:
        """
        Compute temporal features optimized for speech detection.

        Speech has characteristic patterns:
        - Rhythmic movement (3-8 Hz for syllables)
        - Consistent amplitude
        - Regular periodicity

        Args:
            current_landmarks: Current frame's mouth landmarks
            current_mar: Current Mouth Aspect Ratio

        Returns:
            Temporal features including speech-specific patterns (20 features)
        """
        self.landmark_history.append(current_landmarks)
        self.mar_history.append(current_mar)

        # Keep only the required window
        if len(self.landmark_history) > self.temporal_window:
            self.landmark_history.pop(0)
        if len(self.mar_history) > self.temporal_window:
            self.mar_history.pop(0)

        # If we don't have enough history, return zeros
        if len(self.landmark_history) < 2:
            return np.zeros(20)  # Increased to 20 features for speech detection

        # Compute differences
        temporal_features = []

        # === Basic Motion Features (5 features) ===
        prev_landmarks = self.landmark_history[-2]
        current_landmarks = self.landmark_history[-1]

        if prev_landmarks.shape == current_landmarks.shape:
            displacement = current_landmarks - prev_landmarks

            # Mean displacement
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

        # === Speech-Specific Features (15 features) ===

        # 1. Movement velocity and acceleration (3 features)
        if len(self.landmark_history) >= 3:
            # Velocity (displacement rate)
            velocity = np.mean(np.abs(self.landmark_history[-1] - self.landmark_history[-2]))
            prev_velocity = np.mean(np.abs(self.landmark_history[-2] - self.landmark_history[-3]))

            # Acceleration (change in velocity)
            acceleration = velocity - prev_velocity

            temporal_features.extend([velocity, prev_velocity, acceleration])
        else:
            temporal_features.extend([0.0, 0.0, 0.0])

        # 2. MAR-based periodicity (5 features)
        if len(self.mar_history) >= 5:
            mar_array = np.array(self.mar_history)

            # MAR variance (speech has consistent opening/closing)
            mar_variance = np.var(mar_array)
            temporal_features.append(mar_variance)

            # MAR range (difference between max and min)
            mar_range = np.max(mar_array) - np.min(mar_array)
            temporal_features.append(mar_range)

            # MAR rate of change (how fast mouth opens/closes)
            mar_diff = np.abs(np.diff(mar_array))
            mar_roc_mean = np.mean(mar_diff)
            mar_roc_std = np.std(mar_diff)
            temporal_features.extend([mar_roc_mean, mar_roc_std])

            # Zero-crossing rate of MAR changes (indicates rhythmic pattern)
            mar_changes = np.diff(mar_array)
            zero_crossings = np.sum(mar_changes[:-1] * mar_changes[1:] < 0)
            zcr = zero_crossings / max(1, len(mar_changes) - 1)
            temporal_features.append(zcr)
        else:
            temporal_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # 3. Movement consistency (4 features)
        if len(self.landmark_history) >= self.temporal_window:
            # Calculate movement across entire window
            all_displacements = []
            for i in range(1, len(self.landmark_history)):
                disp = np.mean(np.abs(self.landmark_history[i] - self.landmark_history[i-1]))
                all_displacements.append(disp)

            all_displacements = np.array(all_displacements)

            # Movement consistency (low variance = consistent speech pattern)
            movement_std = np.std(all_displacements)
            movement_mean = np.mean(all_displacements)
            temporal_features.extend([movement_mean, movement_std])

            # Movement regularity (coefficient of variation)
            movement_cv = movement_std / (movement_mean + 1e-6)
            temporal_features.append(movement_cv)

            # Sustained movement (ratio of frames with movement)
            movement_threshold = np.median(all_displacements)
            sustained_ratio = np.sum(all_displacements > movement_threshold) / len(all_displacements)
            temporal_features.append(sustained_ratio)
        else:
            temporal_features.extend([0.0, 0.0, 0.0, 0.0])

        # 4. Frequency domain hint (3 features) - simplified without FFT
        if len(self.mar_history) >= 10:
            mar_array = np.array(self.mar_history[-10:])

            # Count peaks (mouth opening cycles)
            mar_diff = np.diff(mar_array)
            peaks = np.sum((mar_diff[:-1] > 0) & (mar_diff[1:] < 0))
            peak_rate = peaks / 10.0  # Normalized by window
            temporal_features.append(peak_rate)

            # Average peak amplitude
            if peaks > 0:
                peak_indices = np.where((mar_diff[:-1] > 0) & (mar_diff[1:] < 0))[0] + 1
                peak_amplitudes = mar_array[peak_indices] - np.min(mar_array)
                avg_peak_amplitude = np.mean(peak_amplitudes)
                peak_amplitude_std = np.std(peak_amplitudes)
            else:
                avg_peak_amplitude = 0.0
                peak_amplitude_std = 0.0

            temporal_features.extend([avg_peak_amplitude, peak_amplitude_std])
        else:
            temporal_features.extend([0.0, 0.0, 0.0])

        # Ensure exactly 20 features
        while len(temporal_features) < 20:
            temporal_features.append(0.0)

        return np.array(temporal_features[:20])

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
        Extract all features for the current frame optimized for speech/talking detection.

        Args:
            frame: Input frame
            mouth_landmarks: Mouth landmark coordinates
            mouth_bbox: Mouth bounding box

        Returns:
            Feature vector of dimension 35 (updated from 25 for speech detection):
            - MAR: 1
            - Inter-landmark distances: 10
            - Temporal features: 20 (increased from 10, includes speech-specific features)
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

        # 3. Temporal features for speech detection (20 features)
        temporal = self.compute_temporal_features(mouth_landmarks, mar)
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
        self.mar_history = []


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
