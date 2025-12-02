import argparse
import json
from pathlib import Path
import sys

# Ensure project root is on the path when running from the scripts directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mouth_movement.training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train lightweight mouth movement detector.")
    parser.add_argument("--metadata", required=True, help="CSV with frame filepaths and binary labels.")
    parser.add_argument(
        "--predictor",
        required=False,
        help="Path to dlib shape_predictor_68_face_landmarks.dat (required if --backend dlib).",
    )
    parser.add_argument(
        "--backend",
        choices=["mediapipe", "dlib"],
        default="mediapipe",
        help="Landmark backend. Default mediapipe avoids dlib build issues.",
    )
    parser.add_argument("--hidden-dim", type=int, default=12, help="Hidden layer width.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--artifacts", default="artifacts", help="Directory to save model and scaler.")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.artifacts).mkdir(parents=True, exist_ok=True)
    metrics = train_model(
        metadata_csv=args.metadata,
        predictor_path=args.predictor,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        l2_lambda=args.l2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        artifacts_dir=args.artifacts,
        backend=args.backend,
    )
    print("Test metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
