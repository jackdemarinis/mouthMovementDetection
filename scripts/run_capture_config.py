import argparse
import json
from pathlib import Path
import sys

import cv2

# Ensure project root is on sys.path when running from the scripts directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run capture_dataset with config values.")
    parser.add_argument("--config", default="config/config.json", nargs="?", help="Path to config JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config or "config/config.json")
    out_dir = cfg.get("capture_out_dir", "data/frames")
    metadata = cfg.get("capture_metadata", "data/metadata.csv")
    source = cfg.get("capture_source", "0")
    every = int(cfg.get("capture_every", 3))
    sleep = float(cfg.get("capture_sleep", 0.05))

    # Directly invoke the capture script logic to avoid re-typing options
    from scripts.capture_dataset import main as capture_main  # type: ignore

    sys.argv = [
        "capture_dataset",
        "--out-dir",
        out_dir,
        "--metadata",
        metadata,
        "--source",
        str(source),
        "--every",
        str(every),
        "--sleep",
        str(sleep),
    ]
    capture_main()


if __name__ == "__main__":
    main()
