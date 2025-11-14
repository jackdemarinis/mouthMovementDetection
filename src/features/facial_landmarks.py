"""
Facial Landmark Detection Module

This module provides face detection and mouth region estimation using OpenCV.
Works with Python 3.13+ without external dependencies like MediaPipe or dlib.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os


class FacialLandmarkDetector:
    """
    Detects faces and estimates mouth region using OpenCV.

    Uses Haar Cascade for face detection and geometric estimation for mouth location.
    This is a lightweight alternative to MediaPipe/dlib that works on all Python versions.
    """

    def __init__(self, detector_type: str = "opencv", confidence_threshold: float = 0.5):
        """
        Initialize the facial landmark detector.

        Args:
            detector_type: Type of detector (only "opencv" supported in this version)
            confidence_threshold: Minimum confidence for detection (not used in OpenCV Haar)
        """
        self.detector_type = "opencv"
        self.confidence_threshold = confidence_threshold

        # Load Haar Cascade classifiers
        cascade_path = cv2.data.haarcascades

        # Face detector
        face_cascade_file = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(face_cascade_file)

        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load face cascade from {face_cascade_file}")

        # Mouth detector (optional - for refinement)
        mouth_cascade_file = os.path.join(cascade_path, 'haarcascade_smile.xml')
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_file)

        print(f"✓ OpenCV face detector initialized")

    def detect_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and generate pseudo-landmarks for mouth region.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            numpy array of shape (N, 2) containing (x, y) coordinates of mouth landmarks,
            or None if no face is detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Use the largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

        fx, fy, fw, fh = faces[0]

        # Generate mouth landmarks based on face geometry
        # Mouth is typically in lower third of face, centered
        mouth_landmarks = self._generate_mouth_landmarks(fx, fy, fw, fh)

        return mouth_landmarks

    def _generate_mouth_landmarks(self, face_x: int, face_y: int,
                                  face_w: int, face_h: int) -> np.ndarray:
        """
        Generate estimated mouth landmarks based on face bounding box.

        Creates ~20 points around the mouth region using facial proportions.

        Args:
            face_x, face_y: Top-left corner of face
            face_w, face_h: Width and height of face

        Returns:
            Array of mouth landmark coordinates
        """
        # Mouth is typically:
        # - Horizontally centered
        # - Starts at about 60% down the face
        # - Width is about 50% of face width
        # - Height is about 15% of face height

        mouth_center_x = face_x + face_w // 2
        mouth_center_y = face_y + int(face_h * 0.7)

        mouth_width = int(face_w * 0.5)
        mouth_height = int(face_h * 0.15)

        # Generate landmarks around mouth perimeter
        landmarks = []

        # Top lip (outer) - 8 points
        for i in range(8):
            t = i / 7  # 0 to 1
            x = mouth_center_x - mouth_width//2 + int(t * mouth_width)
            # Slight curve for top lip
            y = mouth_center_y - mouth_height//2 + int(abs(t - 0.5) * mouth_height * 0.3)
            landmarks.append([x, y])

        # Bottom lip (outer) - 8 points
        for i in range(8):
            t = i / 7
            x = mouth_center_x + mouth_width//2 - int(t * mouth_width)
            # Slight curve for bottom lip
            y = mouth_center_y + mouth_height//2 - int(abs(t - 0.5) * mouth_height * 0.3)
            landmarks.append([x, y])

        # Inner mouth - 4 corner points
        landmarks.append([mouth_center_x - mouth_width//3, mouth_center_y - mouth_height//4])
        landmarks.append([mouth_center_x + mouth_width//3, mouth_center_y - mouth_height//4])
        landmarks.append([mouth_center_x + mouth_width//3, mouth_center_y + mouth_height//4])
        landmarks.append([mouth_center_x - mouth_width//3, mouth_center_y + mouth_height//4])

        return np.array(landmarks, dtype=np.int32)

    def get_mouth_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract mouth region landmarks.

        Args:
            frame: Input frame

        Returns:
            numpy array of mouth landmarks (N, 2) or None
        """
        # In this simplified version, all landmarks are mouth landmarks
        return self.detect_face_landmarks(frame)

    def visualize_landmarks(self, frame: np.ndarray, landmarks: np.ndarray,
                           mouth_only: bool = True) -> np.ndarray:
        """
        Draw landmarks on the frame for visualization.

        Args:
            frame: Input frame
            landmarks: Landmark coordinates
            mouth_only: Not used (all landmarks are mouth landmarks)

        Returns:
            Frame with landmarks drawn
        """
        vis_frame = frame.copy()

        # Draw all landmarks
        for x, y in landmarks:
            cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        # Draw connecting lines for better visualization
        num_points = len(landmarks)
        if num_points >= 16:
            # Connect top lip
            for i in range(7):
                pt1 = tuple(landmarks[i].astype(int))
                pt2 = tuple(landmarks[i+1].astype(int))
                cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 1)

            # Connect bottom lip
            for i in range(8, 15):
                pt1 = tuple(landmarks[i].astype(int))
                pt2 = tuple(landmarks[i+1].astype(int))
                cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 1)

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

        # Add some padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        width += 2 * padding
        height += 2 * padding

        return x_min, y_min, width, height


if __name__ == "__main__":
    # Test the detector
    print("Testing OpenCV-based face and mouth detection...")
    print("This uses Haar Cascades - works on all Python versions!")
    print()

    detector = FacialLandmarkDetector(detector_type="opencv")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit(1)

    print("Press 'q' to quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect landmarks
        landmarks = detector.detect_face_landmarks(frame)

        if landmarks is not None:
            # Visualize
            vis_frame = detector.visualize_landmarks(frame, landmarks)

            # Get mouth region
            mouth_landmarks = detector.get_mouth_landmarks(frame)
            if mouth_landmarks is not None:
                x, y, w, h = detector.get_mouth_bounding_box(mouth_landmarks)
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.putText(vis_frame, "Mouth detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            vis_frame = frame
            cv2.putText(vis_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Facial Landmarks (OpenCV)", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nTest complete!")
