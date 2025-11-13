"""
Data Collection Script for Mouth Movement Detection

Collects video data from webcam with synchronized audio for automatic labeling.
"""

import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime
import json
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.facial_landmarks import FacialLandmarkDetector
from features.mouth_features import MouthFeatureExtractor


class DataCollector:
    """
    Collects training data for mouth movement detection.

    Records video from webcam and saves frames along with metadata.
    Provides visual feedback during recording.
    """

    def __init__(self, output_dir: str = "data/raw", fps: int = 30):
        """
        Initialize the data collector.

        Args:
            output_dir: Directory to save collected data
            fps: Frames per second for video capture
        """
        self.output_dir = output_dir
        self.fps = fps

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize detectors
        self.landmark_detector = FacialLandmarkDetector(detector_type="mediapipe")
        self.feature_extractor = MouthFeatureExtractor()

        # Recording state
        self.is_recording = False
        self.frames = []
        self.features_list = []
        self.timestamps = []

    def record_session(self, duration: int = 60, session_name: str = None):
        """
        Record a data collection session.

        Args:
            duration: Maximum duration in seconds (0 for unlimited)
            session_name: Name for this session (auto-generated if None)
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)

        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        print("\n" + "="*60)
        print("MOUTH MOVEMENT DATA COLLECTION")
        print("="*60)
        print(f"Session: {session_name}")
        print(f"Output: {session_dir}")
        print(f"Duration: {duration}s" if duration > 0 else "Duration: Unlimited")
        print("\nControls:")
        print("  SPACE - Start/Stop recording")
        print("  's' - Save current session")
        print("  'q' - Quit")
        print("\nTips for good data:")
        print("  1. Record yourself speaking normally")
        print("  2. Record periods of silence (mouth closed)")
        print("  3. Try different head angles")
        print("  4. Ensure good lighting")
        print("="*60 + "\n")

        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Check duration
            if duration > 0 and time.time() - start_time > duration:
                print("\nMaximum duration reached. Saving...")
                break

            # Detect landmarks and extract features
            mouth_landmarks = self.landmark_detector.get_mouth_landmarks(frame)

            # Create display frame
            display_frame = frame.copy()

            if mouth_landmarks is not None:
                # Get bounding box
                x, y, w, h = self.landmark_detector.get_mouth_bounding_box(mouth_landmarks)

                # Extract features
                features = self.feature_extractor.extract_all_features(
                    frame, mouth_landmarks, (x, y, w, h)
                )

                # Draw mouth region
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Visualize landmarks
                for lx, ly in mouth_landmarks:
                    cv2.circle(display_frame, (int(lx), int(ly)), 1, (0, 255, 0), -1)

                # If recording, save frame and features
                if self.is_recording:
                    self.frames.append(frame.copy())
                    self.features_list.append(features)
                    self.timestamps.append(time.time() - start_time)
                    frame_count += 1

            else:
                cv2.putText(display_frame, "NO FACE DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display status
            status_color = (0, 0, 255) if self.is_recording else (255, 255, 255)
            status_text = "RECORDING" if self.is_recording else "PAUSED"
            cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Display frame count
            cv2.putText(display_frame, f"Frames: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display elapsed time
            elapsed = int(time.time() - start_time)
            cv2.putText(display_frame, f"Time: {elapsed}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Data Collection", display_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                self.is_recording = not self.is_recording
                print(f"{'Started' if self.is_recording else 'Stopped'} recording")
            elif key == ord('s'):
                print("\nSaving session...")
                self._save_session(session_dir)
                print(f"Saved {len(self.frames)} frames")
                # Reset for next recording
                self.frames = []
                self.features_list = []
                self.timestamps = []
                frame_count = 0
                self.feature_extractor.reset_history()

        # Save remaining data
        if len(self.frames) > 0:
            print("\nSaving final session...")
            self._save_session(session_dir)
            print(f"Saved {len(self.frames)} frames")

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print(f"Data collection complete!")
        print(f"Session saved to: {session_dir}")
        print("="*60 + "\n")

    def _save_session(self, session_dir: str):
        """
        Save collected frames and features.

        Args:
            session_dir: Directory to save the session
        """
        if len(self.frames) == 0:
            print("No frames to save!")
            return

        # Create subdirectories
        frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Save frames
        for i, frame in enumerate(self.frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame)

        # Save features
        features_array = np.array(self.features_list)
        features_path = os.path.join(session_dir, "features.npy")
        np.save(features_path, features_array)

        # Save metadata
        metadata = {
            "num_frames": len(self.frames),
            "fps": self.fps,
            "timestamps": self.timestamps,
            "feature_dim": features_array.shape[1] if len(features_array) > 0 else 0,
            "collection_date": datetime.now().isoformat()
        }

        metadata_path = os.path.join(session_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(self.frames)} frames to {frames_dir}")
        print(f"Saved features to {features_path}")
        print(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect training data for mouth movement detection")
    parser.add_argument("--output", type=str, default="data/raw",
                       help="Output directory for collected data")
    parser.add_argument("--duration", type=int, default=0,
                       help="Maximum recording duration in seconds (0 for unlimited)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    parser.add_argument("--session", type=str, default=None,
                       help="Session name (auto-generated if not provided)")

    args = parser.parse_args()

    collector = DataCollector(output_dir=args.output, fps=args.fps)
    collector.record_session(duration=args.duration, session_name=args.session)


if __name__ == "__main__":
    main()
