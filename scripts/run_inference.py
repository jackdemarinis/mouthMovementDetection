import argparse
import json
import sys
from pathlib import Path

import cv2

# Ensure project root is on the path when running from the scripts directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mouth_movement.inference import MouthMovementDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time mouth movement inference.")
    parser.add_argument(
        "--config",
        default="config/config.json",
        nargs="?",
        help="Path to config JSON (defaults to config/config.json).",
    )
    parser.add_argument("--predictor", help="Path to dlib shape_predictor_68_face_landmarks.dat (if backend=dlib).")
    parser.add_argument("--backend", choices=["mediapipe", "dlib"], help="Landmark backend.")
    parser.add_argument("--model", help="Path to trained model .npz")
    parser.add_argument("--scaler", help="Path to saved scaler .npy")
    parser.add_argument("--source", help="Video source (camera index or file path).")
    parser.add_argument("--threshold", type=float, help="Classification threshold.")
    parser.add_argument("--smooth", type=int, help="Sliding window size for probability smoothing.")
    return parser.parse_args()


def merge_config(args):
    cfg = {}
    cfg_path = args.config or "config/config.json"
    path_obj = Path(cfg_path)
    if path_obj.exists():
        with open(path_obj, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    # CLI overrides config
    def pick(key, default=None):
        val = getattr(args, key, None)
        return val if val is not None else cfg.get(key, default)

    out = {
        "predictor": pick("predictor"),
        "backend": pick("backend", "mediapipe"),
        "model": pick("model", "artifacts/mouth_movement_nn.npz"),
        "scaler": pick("scaler", "artifacts/feature_scaler.npy"),
        "source": pick("source", "0"),
        "threshold": float(pick("threshold", 0.5)),
        "smooth": int(pick("smooth", 5)),
    }
    return out


def main():
    args = parse_args()
    cfg = merge_config(args)
    src_val = cfg["source"]
    src = int(src_val) if isinstance(src_val, str) and src_val.isdigit() else src_val
    detector = MouthMovementDetector(
        model_path=cfg["model"],
        scaler_path=cfg["scaler"],
        predictor_path=cfg["predictor"],
        threshold=cfg["threshold"],
        smooth_window=cfg["smooth"],
        backend=cfg["backend"],
    )

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {cfg['source']}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        prob, detection = detector.predict_frame(frame)
        annotated = detector.annotate_frame(frame, prob, detection)
        cv2.imshow("Mouth Movement", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
