import argparse
import json
import random
from itertools import cycle
from pathlib import Path
from typing import Tuple

import cv2
import pandas as pd


def load_config(default_cfg: str) -> dict:
    cfg_path = Path(default_cfg)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Balance positives by augmenting talking frames.")
    parser.add_argument("--metadata", help="Path to metadata CSV (filepath,label). Defaults to config or data/metadata.csv.")
    parser.add_argument("--max_factor", type=float, default=2.0, help="Cap augmented positives at this multiple of original positives.")
    parser.add_argument("--config", default="config/config.json", nargs="?", help="Config JSON path (optional).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def augment_image(img) -> None:
    # random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    # color jitter in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    sat_scale = random.uniform(0.8, 1.3)
    val_scale = random.uniform(0.8, 1.2)
    hsv[:, :, 1] *= sat_scale
    hsv[:, :, 2] *= val_scale
    hsv[:, :, 1] = hsv[:, :, 1].clip(0, 255)
    hsv[:, :, 2] = hsv[:, :, 2].clip(0, 255)
    img_out = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return img_out


def main():
    args = parse_args()
    cfg = load_config(args.config or "config/config.json")
    meta_path = Path(args.metadata or cfg.get("metadata", "data/metadata.csv"))
    random.seed(args.seed)

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata CSV not found: {meta_path}")

    df = pd.read_csv(meta_path)
    if not {"filepath", "label"}.issubset(df.columns):
        raise ValueError("metadata CSV must have columns: filepath,label")

    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    pos_count, neg_count = len(pos_df), len(neg_df)

    if pos_count >= neg_count:
        print(f"No augmentation needed. positives={pos_count} negatives={neg_count}")
        return

    target = min(int(pos_count * args.max_factor), neg_count)
    need = max(0, target - pos_count)
    print(f"Augmenting positives: current {pos_count}, negatives {neg_count}, target positives {target}, need {need}.")

    rows = []
    for i, (_, row) in zip(range(need), cycle(pos_df.iterrows())):
        img_path = Path(row["filepath"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        aug = augment_image(img)
        out_dir = img_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        new_name = f"{img_path.stem}_aug_{i:05d}{img_path.suffix}"
        out_path = out_dir / new_name
        cv2.imwrite(str(out_path), aug)
        rows.append({"filepath": str(out_path).replace("\\", "/"), "label": 1})

    if not rows:
        raise RuntimeError("No augmented images were created; check source image readability.")

    df_aug = pd.DataFrame(rows)
    df_final = pd.concat([df, df_aug], ignore_index=True)
    df_final.to_csv(meta_path, index=False)
    print(f"Augmented {len(rows)} images. New counts -> positives={len(df_final[df_final.label==1])}, negatives={len(df_final[df_final.label==0])}")


if __name__ == "__main__":
    main()
