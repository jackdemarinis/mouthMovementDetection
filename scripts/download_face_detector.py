import argparse
from pathlib import Path

import requests


PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_models/master/res10_300x300_ssd_iter_140000.caffemodel"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main():
    parser = argparse.ArgumentParser(description="Download OpenCV DNN face detector model.")
    parser.add_argument("--out-dir", default="models", help="Directory to save model files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    proto_path = out_dir / "deploy.prototxt"
    model_path = out_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    if not proto_path.exists():
        print("Downloading prototxt...")
        download(PROTO_URL, proto_path)
    else:
        print("Prototxt already exists.")

    if not model_path.exists():
        print("Downloading caffemodel...")
        download(MODEL_URL, model_path)
    else:
        print("Caffemodel already exists.")

    print(f"Saved to {proto_path} and {model_path}")


if __name__ == "__main__":
    main()
