import argparse
import csv
import os
from pathlib import Path
import time

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Capture labeled frames from webcam for mouth movement dataset.")
    p.add_argument("--out-dir", default="data/frames", help="Directory to save captured frames.")
    p.add_argument("--metadata", default="data/metadata.csv", help="Path to write CSV labels.")
    p.add_argument("--source", default="0", help="Camera index or video path.")
    p.add_argument("--every", type=int, default=5, help="Capture every Nth frame while recording (higher = lower FPS).")
    p.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between frames to slow capture/display.")
    return p.parse_args()


def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.metadata)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source {args.source}")

    frame_idx = 0
    save_idx = 0
    current_label = 0  # 0 = not talking, 1 = talking
    capture = False

    print("Controls: 't' toggle talking/not talking label, 'r' start/stop capture, 'q' quit.")
    print(f"Saving frames to: {out_dir}  metadata: {meta_path}")

    meta_file = open(meta_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(meta_file)
    writer.writerow(["filepath", "label"])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            label_text = "talking (1)" if current_label == 1 else "not talking (0)"
            status = "REC" if capture else "IDLE"
            cv2.putText(frame, f"Label: {label_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if capture else (255, 255, 0), 2)
            cv2.imshow("Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                current_label = 1 - current_label
            elif key == ord("r"):
                capture = not capture

            if capture and frame_idx % args.every == 0:
                fname = f"frame_{int(time.time())}_{save_idx:05d}.jpg"
                fpath = out_dir / fname
                cv2.imwrite(str(fpath), frame)
                writer.writerow([str(fpath).replace("\\", "/"), current_label])
                save_idx += 1

            frame_idx += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        meta_file.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
