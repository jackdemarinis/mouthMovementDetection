"""
Facial Landmark Detection Module

This module provides face detection and facial landmark localization
using MediaPipe and dlib.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List


class FacialLandmarkDetector:
    """
    Detects faces and extracts facial landmarks using MediaPipe or dlib.

    MediaPipe provides 468 facial landmarks, but we focus on the mouth region
    (landmarks 61-291 specifically around lips).
    """

    def __init__(self, detector_type: str = "mediapipe", confidence_threshold: float = 0.5):
        """
        Initialize the facial landmark detector.

        Args:
            detector_type: Type of detector ("mediapipe" or "dlib")
            confidence_threshold: Minimum confidence for detection
        """
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold

        if detector_type == "mediapipe":
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=confidence_threshold,
                min_tracking_confidence=confidence_threshold
            )
            # MediaPipe mouth landmark indices (inner and outer lips)
            self.MOUTH_INDICES = [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375,  # Upper outer lip
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415,   # Lower outer lip
                95, 88, 178, 87, 14, 317, 402, 318, 324, 308,  # Upper inner lip
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324    # Lower inner lip
            ]
        elif detector_type == "dlib":
            try:
                import dlib
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
                # dlib mouth landmark indices (48-67)
                self.MOUTH_INDICES = list(range(48, 68))
            except Exception as e:
                raise ImportError(f"Failed to initialize dlib: {e}. Please install dlib or use MediaPipe.")
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    def detect_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in a frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            numpy array of shape (N, 2) containing (x, y) coordinates of landmarks,
            or None if no face is detected
        """
        if self.detector_type == "mediapipe":
            return self._detect_mediapipe(frame)
        elif self.detector_type == "dlib":
            return self._detect_dlib(frame)

    def _detect_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect landmarks using MediaPipe."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]

        # Extract all landmarks
        h, w = frame.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])

        return np.array(landmarks)

    def _detect_dlib(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect landmarks using dlib."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        # Get landmarks for the first face
        face = faces[0]
        shape = self.predictor(gray, face)

        # Convert to numpy array
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        return landmarks

    def get_mouth_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract only mouth region landmarks.

        Args:
            frame: Input frame

        Returns:
            numpy array of mouth landmarks (N, 2) or None
        """
        all_landmarks = self.detect_face_landmarks(frame)

        if all_landmarks is None:
            return None

        # Extract mouth landmarks based on indices
        mouth_landmarks = all_landmarks[self.MOUTH_INDICES]

        return mouth_landmarks

    def visualize_landmarks(self, frame: np.ndarray, landmarks: np.ndarray,
                           mouth_only: bool = False) -> np.ndarray:
        """
        Draw landmarks on the frame for visualization.

        Args:
            frame: Input frame
            landmarks: Landmark coordinates
            mouth_only: If True, only draw mouth landmarks

        Returns:
            Frame with landmarks drawn
        """
        vis_frame = frame.copy()

        if mouth_only and self.detector_type == "mediapipe":
            # Draw only mouth landmarks
            for idx in self.MOUTH_INDICES:
                if idx < len(landmarks):
                    x, y = landmarks[idx]
                    cv2.circle(vis_frame, (int(x), int(y)), 1, (0, 255, 0), -1)
        else:
            # Draw all landmarks
            for x, y in landmarks:
                cv2.circle(vis_frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        return vis_frame

    def get_mouth_bounding_box(self, mouth_landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box around mouth region.

        Args:
            mouth_landmarks: Mouth landmark coordinates

        Returns:
            Tuple of (x, y, width, height)
        """
        x_min = int(np.min(mouth_landmarks[:, 0]))
        y_min = int(np.min(mouth_landmarks[:, 1]))
        x_max = int(np.max(mouth_landmarks[:, 0]))
        y_max = int(np.max(mouth_landmarks[:, 1]))

        width = x_max - x_min
        height = y_max - y_min

        return x_min, y_min, width, height

    def __del__(self):
        """Cleanup resources."""
        if self.detector_type == "mediapipe" and hasattr(self, 'face_mesh'):
            self.face_mesh.close()


if __name__ == "__main__":
    # Test the detector
    detector = FacialLandmarkDetector(detector_type="mediapipe")

    # Open webcam
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect landmarks
        landmarks = detector.detect_face_landmarks(frame)

        if landmarks is not None:
            # Visualize
            vis_frame = detector.visualize_landmarks(frame, landmarks, mouth_only=True)

            # Get mouth region
            mouth_landmarks = detector.get_mouth_landmarks(frame)
            if mouth_landmarks is not None:
                x, y, w, h = detector.get_mouth_bounding_box(mouth_landmarks)
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            vis_frame = frame
            cv2.putText(vis_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Facial Landmarks", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
