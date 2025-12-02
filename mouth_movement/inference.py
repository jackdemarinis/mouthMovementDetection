from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from .data import load_scaler
from .features import MouthFeatureExtractor, draw_detection, load_dlib_predictor
from .model import FeedforwardNN


class MouthMovementDetector:
    """
    Real-time mouth movement detector that combines the feature extractor,
    fitted scaler, and trained neural network.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        predictor_path: Optional[str] = None,
        threshold: float = 0.5,
        smooth_window: int = 5,
        backend: str = "mediapipe",
    ) -> None:
        if backend == "dlib":
            if predictor_path is None:
                raise ValueError("predictor_path is required when backend='dlib'")
            detector, predictor = load_dlib_predictor(predictor_path)
            self.extractor = MouthFeatureExtractor(detector, predictor, use_mediapipe=False)
        else:
            self.extractor = MouthFeatureExtractor(detector=None, predictor=None, use_mediapipe=True)
        self.scaler = load_scaler(scaler_path)
        self.model = FeedforwardNN.load(model_path)
        self.threshold = threshold
        self.window: Deque[float] = deque(maxlen=smooth_window)

    def predict_frame(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
        feat, detection = self.extractor.extract_features(frame)
        if feat is None:
            return None, None

        feat_norm = self.scaler.transform(feat.reshape(1, -1))
        prob = float(self.model.predict_proba(feat_norm)[0][0])
        self.window.append(prob)
        smooth_prob = float(np.mean(self.window))
        return smooth_prob, detection

    def annotate_frame(self, frame: np.ndarray, prob: Optional[float], detection) -> np.ndarray:
        view = frame.copy()
        if detection is not None:
            view = draw_detection(view, detection)
        if prob is not None:
            label = "moving" if prob >= self.threshold else "not moving"
            cv2.putText(
                view,
                f"{label}: {prob:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if label == "moving" else (0, 0, 255),
                2,
            )
        return view
