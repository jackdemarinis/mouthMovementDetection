import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import dlib
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

# Landmark indices for the mouth in the 68-point dlib model
MOUTH_START = 48
MOUTH_END = 68


@dataclass
class LandmarkDetection:
    points: np.ndarray  # shape: (68, 2) in (x, y) pixel coordinates
    bbox: Tuple[int, int, int, int]  # x, y, w, h


def load_dlib_predictor(predictor_path: str):
    """
    Convenience helper to load the dlib frontal face detector and the 68-point
    landmark predictor.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor


def load_opencv_dnn_face_detector(
    proto_path: str = "models/deploy.prototxt",
    model_path: str = "models/res10_300x300_ssd_iter_140000.caffemodel",
) -> Optional[cv2.dnn.Net]:
    """
    Load the OpenCV DNN face detector (Res10 SSD). Returns None if files are missing.
    """
    if not (cv2.os.path.exists(proto_path) and cv2.os.path.exists(model_path)):
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        return net
    except Exception:
        return None


def _rect_to_bbox(rect: dlib.rectangle) -> Tuple[int, int, int, int]:
    return rect.left(), rect.top(), rect.width(), rect.height()


def _shape_to_np(shape: dlib.full_object_detection) -> np.ndarray:
    coords = np.zeros((shape.num_parts, 2), dtype=np.float32)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    # Use outer mouth landmarks for MAR
    A = np.linalg.norm(mouth[13] - mouth[19])  # 61-67
    B = np.linalg.norm(mouth[15] - mouth[17])  # 63-65
    C = np.linalg.norm(mouth[12] - mouth[16])  # 60-64
    return (A + B) / (2.0 * C + 1e-6)


class MouthFeatureExtractor:
    """
    Extracts geometry, temporal, and intensity-based features from the mouth
    region given a video frame.
    """

    def __init__(
        self,
        detector: Optional[dlib.fhog_object_detector] = None,
        predictor: Optional[dlib.shape_predictor] = None,
        temporal_weight: float = 0.7,
        dnn_net: Optional[cv2.dnn.Net] = None,
        dnn_thresh: float = 0.5,
        use_mediapipe: bool = True,
    ) -> None:
        self.detector = detector
        self.predictor = predictor
        self.temporal_weight = temporal_weight
        self._prev_mouth: Optional[np.ndarray] = None
        self.dnn_net = dnn_net
        self.dnn_thresh = dnn_thresh
        self.use_mediapipe = use_mediapipe and mp is not None
        self.mp_mesh = None
        if self.use_mediapipe:
            # static_image_mode=True to maximize detection quality on single frames
            self.mp_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def reset_temporal_state(self) -> None:
        self._prev_mouth = None

    def detect(self, frame: np.ndarray) -> Optional[LandmarkDetection]:
        # Normalize frame to uint8 BGR before grayscale conversion
        try:
            if frame.dtype != np.uint8:
                frame = cv2.convertScaleAbs(frame)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                pass
            else:
                return None
        except Exception:
            return None

        h, w = frame.shape[:2]

        # Preferred: MediaPipe face mesh for reliable landmarks
        if self.mp_mesh is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.mp_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    lip_indices = [
                        61, 185, 40, 39, 37, 0, 267, 270, 409, 291,
                        375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
                    ]
                    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in lip_indices], dtype=np.float32)
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                    return LandmarkDetection(points=pts, bbox=bbox)
            except Exception:
                pass

        # Fallback: DNN face detector + dlib predictor if available
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces: List[dlib.rectangle] = []
        if self.dnn_net is not None:
            try:
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
                self.dnn_net.setInput(blob)
                detections = self.dnn_net.forward()
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence >= self.dnn_thresh:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w - 1, x2)
                        y2 = min(h - 1, y2)
                        if x2 > x1 and y2 > y1:
                            faces.append(dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))
            except Exception:
                faces = []

        # Fallback to dlib HOG detector if still nothing
        if not faces and self.detector is not None:
            try:
                faces = self.detector(gray, 2)
            except Exception:
                faces = []

        if not faces:
            # Final fallback: Haar cascade
            haar_path = getattr(cv2.data, "haarcascades", "")
            if haar_path:
                cascade = cv2.CascadeClassifier(haar_path + "haarcascade_frontalface_default.xml")
                if not cascade.empty():
                    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
                    if len(boxes) > 0:
                        faces = [dlib.rectangle(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in boxes]

        if not faces or self.predictor is None:
            return None

        best = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, best)
        points68 = _shape_to_np(shape)
        mouth = points68[MOUTH_START:MOUTH_END]
        return LandmarkDetection(points=mouth, bbox=_rect_to_bbox(best))

    def _compute_temporal(self, mouth_points: np.ndarray) -> Tuple[float, float, float]:
        if self._prev_mouth is None:
            self._prev_mouth = mouth_points.copy()
            return 0.0, 0.0, 0.0

        deltas = mouth_points - self._prev_mouth
        # Exponential smoothing to dampen jitter
        self._prev_mouth = (
            self.temporal_weight * mouth_points
            + (1.0 - self.temporal_weight) * self._prev_mouth
        )
        magnitudes = np.linalg.norm(deltas, axis=1)
        return float(np.mean(magnitudes)), float(np.std(magnitudes)), float(np.max(magnitudes))

    def _compute_patch_stats(
        self, frame: np.ndarray, mouth_points: np.ndarray
    ) -> Tuple[float, float, float, float]:
        x_min = int(np.clip(np.min(mouth_points[:, 0]) - 3, 0, frame.shape[1] - 1))
        y_min = int(np.clip(np.min(mouth_points[:, 1]) - 3, 0, frame.shape[0] - 1))
        x_max = int(np.clip(np.max(mouth_points[:, 0]) + 3, 0, frame.shape[1] - 1))
        y_max = int(np.clip(np.max(mouth_points[:, 1]) + 3, 0, frame.shape[0] - 1))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        mean_val = float(np.mean(roi))
        std_val = float(np.std(roi))
        grad_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        return mean_val, std_val, float(np.mean(magnitude)), float(np.std(magnitude))

    def _inter_landmark_distances(self, mouth: np.ndarray) -> List[float]:
        pairs = [
            (0, 6),  # corners
            (2, 10),
            (4, 8),
            (12, 16),  # inner top-bottom
            (14, 18),
        ]
        dists = []
        for a, b in pairs:
            dists.append(float(np.linalg.norm(mouth[a] - mouth[b])))
        return dists

    def _normalized_key_points(self, mouth: np.ndarray) -> List[float]:
        x_min, y_min = np.min(mouth, axis=0)
        x_max, y_max = np.max(mouth, axis=0)
        w = max(x_max - x_min, 1e-3)
        h = max(y_max - y_min, 1e-3)
        key_indices = [0, 3, 6, 9, 12, 15, 16, 18]  # sample key lip points
        subset = mouth[key_indices]
        normed = (subset - np.array([x_min, y_min])) / np.array([w, h])
        return normed.flatten().tolist()

    def extract_features(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[LandmarkDetection]]:
        """
        Returns:
            feature_vector (np.ndarray): shape (N_features,)
            detection (LandmarkDetection)
        """
        detection = self.detect(frame)
        if detection is None:
            return None, None

        mouth = detection.points
        if mouth.shape[0] < 4:
            return None, None

        width = float(np.max(mouth[:, 0]) - np.min(mouth[:, 0]))
        height = float(np.max(mouth[:, 1]) - np.min(mouth[:, 1]))
        mar = height / (width + 1e-6)
        area = float(cv2.contourArea(cv2.convexHull(mouth.astype(np.float32))))
        lip_y_std = float(np.std(mouth[:, 1]))
        tmp_mean, tmp_std, tmp_max = self._compute_temporal(mouth)
        pix_mean, pix_std, edge_mean, edge_std = self._compute_patch_stats(frame, mouth)

        dists = self._inter_landmark_distances(mouth)
        norm_keypoints = self._normalized_key_points(mouth)

        bbox_w, bbox_h = detection.bbox[2], detection.bbox[3]
        bbox_ratio = bbox_w / (bbox_h + 1e-6)
        centroid = np.mean(mouth, axis=0)

        features = [
            mar,
            width,
            height,
            area,
            lip_y_std,
            tmp_mean,
            tmp_std,
            tmp_max,
            pix_mean,
            pix_std,
            edge_mean,
            edge_std,
            centroid[0],
            centroid[1],
            bbox_w,
            bbox_h,
            bbox_ratio,
        ]
        features.extend(dists)
        features.extend(norm_keypoints)

        return np.array(features, dtype=np.float32), detection


def draw_detection(frame: np.ndarray, detection: LandmarkDetection, mouth_only: bool = True) -> np.ndarray:
    """
    Overlay detected landmarks for visualization.
    """
    out = frame.copy()
    points = detection.points if mouth_only else detection.points
    for (x, y) in points.astype(np.int32):
        cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
    x, y, w, h = detection.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return out
