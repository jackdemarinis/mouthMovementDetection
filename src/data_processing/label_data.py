"""
Data Labeling Script

Generates binary labels for collected data through manual annotation.
Labels indicate TALKING/SPEAKING (not just general mouth movement).
"""

import numpy as np
import argparse
import os
import json
import cv2
from typing import List, Tuple, Optional


class DataLabeler:
    """
    Labels speech/talking data for training (VISUAL-ONLY).

    Supports:
    - Manual labeling through GUI (RECOMMENDED)
    - Auto-labeling based on temporal patterns (fallback)
    - Label smoothing and validation

    Labels:
    - 1 = TALKING/SPEAKING
    - 0 = NOT TALKING (silent or non-speech mouth movement)

    IMPORTANT: This is a VISUAL-ONLY system. Labels must be based on visual
    observation of speech, not audio.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the data labeler.

        Args:
            data_dir: Directory containing collected data
        """
        self.data_dir = data_dir
        self.current_labels = []
        self.current_frame_idx = 0

    def manual_labeling(self, session_dir: str, output_path: str = None):
        """
        Manually label frames through GUI.

        Args:
            session_dir: Session directory to label
            output_path: Path to save labels (default: session_dir/labels.npy)
        """
        if output_path is None:
            output_path = os.path.join(session_dir, "labels.npy")

        # Load frames
        frames_dir = os.path.join(session_dir, "frames")
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

        if len(frame_files) == 0:
            print(f"No frames found in {frames_dir}")
            return

        print("\n" + "="*60)
        print("MANUAL LABELING MODE")
        print("="*60)
        print(f"Session: {session_dir}")
        print(f"Frames: {len(frame_files)}")
        print("\nControls:")
        print("  '1' or '→' - Label as TALKING/SPEAKING")
        print("  '0' or '←' - Label as NOT TALKING (silent)")
        print("  'SPACE' - Toggle label")
        print("  's' - Save and continue")
        print("  'q' - Save and quit")
        print("  'b' - Go back to previous frame")
        print("\nIMPORTANT: Label '1' only when actually SPEAKING,")
        print("           not for other mouth movements like yawning or chewing!")
        print("="*60 + "\n")

        # Initialize labels (if existing labels, load them)
        if os.path.exists(output_path):
            self.current_labels = list(np.load(output_path))
            print(f"Loaded existing labels: {len(self.current_labels)} frames")
        else:
            self.current_labels = [-1] * len(frame_files)  # -1 = unlabeled

        self.current_frame_idx = 0

        # Find first unlabeled frame
        for i, label in enumerate(self.current_labels):
            if label == -1:
                self.current_frame_idx = i
                break

        while self.current_frame_idx < len(frame_files):
            # Load and display frame
            frame_path = os.path.join(frames_dir, frame_files[self.current_frame_idx])
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Error loading frame: {frame_path}")
                self.current_frame_idx += 1
                continue

            # Create display
            display = frame.copy()

            # Show current label
            current_label = self.current_labels[self.current_frame_idx]
            if current_label == 1:
                label_text = "TALKING"
                label_color = (0, 255, 0)
            elif current_label == 0:
                label_text = "NOT TALKING"
                label_color = (0, 0, 255)
            else:
                label_text = "UNLABELED"
                label_color = (128, 128, 128)

            cv2.putText(display, label_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

            # Show progress
            progress = f"{self.current_frame_idx + 1}/{len(frame_files)}"
            cv2.putText(display, progress, (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show labeled count
            labeled_count = sum(1 for l in self.current_labels if l != -1)
            labeled_text = f"Labeled: {labeled_count}/{len(frame_files)}"
            cv2.putText(display, labeled_text, (10, display.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Manual Labeling", display)

            # Handle input
            key = cv2.waitKey(0) & 0xFF

            if key == ord('1') or key == 83:  # '1' or right arrow
                self.current_labels[self.current_frame_idx] = 1
                print(f"Frame {self.current_frame_idx}: TALKING")
                self.current_frame_idx += 1

            elif key == ord('0') or key == 81:  # '0' or left arrow
                self.current_labels[self.current_frame_idx] = 0
                print(f"Frame {self.current_frame_idx}: NOT TALKING")
                self.current_frame_idx += 1

            elif key == ord(' '):  # Space - toggle
                if self.current_labels[self.current_frame_idx] == 1:
                    self.current_labels[self.current_frame_idx] = 0
                elif self.current_labels[self.current_frame_idx] == 0:
                    self.current_labels[self.current_frame_idx] = 1
                else:
                    self.current_labels[self.current_frame_idx] = 1
                print(f"Frame {self.current_frame_idx}: Toggled to {self.current_labels[self.current_frame_idx]}")

            elif key == ord('b'):  # Back
                if self.current_frame_idx > 0:
                    self.current_frame_idx -= 1
                    print(f"Back to frame {self.current_frame_idx}")

            elif key == ord('s'):  # Save
                self._save_labels(output_path)
                print(f"Saved labels to {output_path}")

            elif key == ord('q'):  # Quit
                self._save_labels(output_path)
                print(f"Saved labels and quitting...")
                break

        cv2.destroyAllWindows()

        # Final save
        self._save_labels(output_path)
        print("\n" + "="*60)
        print(f"Labeling complete!")
        print(f"Total frames: {len(frame_files)}")
        print(f"Labeled: {labeled_count}/{len(frame_files)}")
        print(f"Labels saved to: {output_path}")
        print("="*60 + "\n")

    def auto_label_from_features(self, session_dir: str, output_path: str = None,
                                 threshold_percentile: float = 60):
        """
        Automatically label frames based on feature variance.

        WARNING: This method cannot reliably distinguish speech from other mouth movements!
        Manual labeling is STRONGLY RECOMMENDED for accurate speech detection.

        Frames with high temporal variance in features are likely "moving".

        Args:
            session_dir: Session directory to label
            output_path: Path to save labels
            threshold_percentile: Percentile threshold for classification
        """
        if output_path is None:
            output_path = os.path.join(session_dir, "labels.npy")

        # Load features
        features_path = os.path.join(session_dir, "features.npy")
        if not os.path.exists(features_path):
            print(f"Features not found: {features_path}")
            return

        features = np.load(features_path)

        print(f"Loaded features: {features.shape}")

        # Compute temporal variance (frame-to-frame changes)
        temporal_diff = np.abs(np.diff(features, axis=0))
        temporal_variance = np.mean(temporal_diff, axis=1)

        # Add zero for first frame
        temporal_variance = np.concatenate([[0], temporal_variance])

        # Threshold based on percentile
        threshold = np.percentile(temporal_variance, threshold_percentile)

        # Label: 1 if variance > threshold, 0 otherwise
        labels = (temporal_variance > threshold).astype(int)

        # Smooth labels (remove single-frame blips)
        labels = self._smooth_labels(labels, window_size=3)

        # Save
        np.save(output_path, labels)

        print("\n" + "="*60)
        print(f"Auto-labeling complete!")
        print(f"Total frames: {len(labels)}")
        print(f"Talking: {np.sum(labels == 1)} ({np.mean(labels) * 100:.1f}%)")
        print(f"Not talking: {np.sum(labels == 0)} ({(1 - np.mean(labels)) * 100:.1f}%)")
        print(f"Threshold: {threshold:.3f}")
        print(f"Labels saved to: {output_path}")
        print("\nWARNING: Auto-labeling cannot distinguish speech from other movements!")
        print("Please review and correct labels manually for best accuracy.")
        print("="*60 + "\n")

    def _smooth_labels(self, labels: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Smooth labels using majority voting in a sliding window.

        Args:
            labels: Raw labels
            window_size: Window size for smoothing

        Returns:
            Smoothed labels
        """
        smoothed = labels.copy()

        for i in range(len(labels)):
            start = max(0, i - window_size // 2)
            end = min(len(labels), i + window_size // 2 + 1)
            window = labels[start:end]

            # Majority vote
            smoothed[i] = 1 if np.sum(window) > len(window) / 2 else 0

        return smoothed

    def _save_labels(self, output_path: str):
        """Save labels to file."""
        labels_array = np.array(self.current_labels)
        np.save(output_path, labels_array)

        # Also save as JSON for readability
        json_path = output_path.replace('.npy', '.json')
        label_dict = {
            "labels": self.current_labels,
            "num_frames": len(self.current_labels),
            "num_talking": int(np.sum(np.array(self.current_labels) == 1)),
            "num_not_talking": int(np.sum(np.array(self.current_labels) == 0)),
            "num_unlabeled": int(np.sum(np.array(self.current_labels) == -1))
        }

        with open(json_path, 'w') as f:
            json.dump(label_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Label speech/talking data (VISUAL-ONLY)")
    parser.add_argument("--session", type=str, required=True,
                       help="Session directory to label")
    parser.add_argument("--mode", type=str, choices=["manual", "auto"], default="manual",
                       help="Labeling mode: manual (recommended) or auto (fallback)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for labels (default: session_dir/labels.npy)")
    parser.add_argument("--threshold", type=float, default=60,
                       help="Threshold percentile for feature-based auto-labeling (default: 60)")

    args = parser.parse_args()

    labeler = DataLabeler(data_dir=os.path.dirname(args.session))

    if args.mode == "manual":
        labeler.manual_labeling(args.session, args.output)
    elif args.mode == "auto":
        print("\nWARNING: Auto-labeling mode cannot distinguish speech from other movements!")
        print("Manual labeling is STRONGLY recommended for accurate speech detection.\n")
        labeler.auto_label_from_features(args.session, args.output, args.threshold)


if __name__ == "__main__":
    main()
