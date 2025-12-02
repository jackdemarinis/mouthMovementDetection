import argparse
import json
from pathlib import Path
import sys

import cv2

# Ensure project root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mouth_movement.inference import MouthMovementDetector
from mouth_movement.training import train_model


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_train(cfg: dict):
    return train_model(
        metadata_csv=cfg["metadata"],
        predictor_path=cfg.get("predictor"),
        hidden_dim=cfg.get("hidden_dim", 12),
        learning_rate=cfg.get("lr", 0.05),
        l2_lambda=cfg.get("l2", 1e-4),
        epochs=cfg.get("epochs", 200),
        batch_size=cfg.get("batch_size", 32),
        artifacts_dir=cfg.get("artifacts", "artifacts"),
        backend=cfg.get("backend", "mediapipe"),
        pos_weight=cfg.get("pos_weight", 1.0),
    )


def run_inference(cfg: dict):
    src = cfg.get("source", "0")
    src = int(src) if isinstance(src, str) and src.isdigit() else src
    detector = MouthMovementDetector(
        model_path=cfg["model_path"],
        scaler_path=cfg["scaler_path"],
        predictor_path=cfg.get("predictor"),
        threshold=cfg.get("threshold", 0.5),
        smooth_window=cfg.get("smooth", 5),
        backend=cfg.get("backend", "mediapipe"),
    )
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {cfg.get('source')}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            prob, detection = detector.predict_frame(frame)
            annotated = detector.annotate_frame(frame, prob, detection)
            cv2.imshow("Mouth Movement", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run training or inference via JSON config.")
    parser.add_argument(
        "--config",
        default="config/config.json",
        nargs="?",
        help="Path to config JSON (defaults to config/config.json).",
    )
    args = parser.parse_args()

    cfg_path = args.config or "config/config.json"
    cfg = load_config(cfg_path)
    mode = cfg.get("mode", "train")
    if mode == "train":
        metrics = run_train(cfg)
        print("Test metrics:", json.dumps(metrics, indent=2))
    elif mode == "inference":
        run_inference(cfg)
    else:
        raise ValueError("mode must be 'train' or 'inference'")


if __name__ == "__main__":
    main()
