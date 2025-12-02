# Mouth Movement Detection (talking vs not talking)

Lightweight pipeline for classifying talking vs not talking from facial landmarks. Includes feature extraction, NumPy feedforward NN, training, webcam inference, and a simple capture tool.

## Prerequisites
- Python 3.10 or 3.11
- CMake 3.27.9 (installed via requirements)
- Mediapipe (installed via requirements) is the default landmark backend.
- Optional dlib 68-point model if you want to use backend=dlib: place `models/shape_predictor_68_face_landmarks.dat` (from https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
- OpenCV DNN face detector fallback: `python scripts/download_face_detector.py` to fetch `models/deploy.prototxt` and `models/res10_300x300_ssd_iter_140000.caffemodel`.

## Install
```powershell
uv python install 3.11
uv venv --python 3.11 .venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

## Capture labeled data (webcam)
Labels are talking (1) vs not talking (0).
```powershell
python scripts/capture_dataset.py --out-dir data/frames --metadata data/metadata.csv --source 0 --every 3 --sleep 0.05
```
Controls: `t` toggle label, `r` start/stop saving every Nth frame, `q` quit.

## Train
`metadata.csv` must contain `filepath,label` rows pointing to real images. Use absolute paths if needed.
```powershell
python scripts/train.py --metadata data/metadata.csv --hidden-dim 12 --lr 0.05 --epochs 200 --batch-size 32 --artifacts artifacts --backend mediapipe
```
Outputs to `artifacts/`: `mouth_movement_nn.npz`, `feature_scaler.npy`, `training_history.json`.
If you want to force dlib landmarks instead of mediapipe, add `--backend dlib --predictor models\shape_predictor_68_face_landmarks.dat`.

## Inference (webcam or video)
- Using config defaults (no extra args):
```powershell
python scripts/run_inference.py
```
- To override, edit `config/config.json` (model, scaler, source, threshold, backend). Press `q` to quit. For dlib, set backend and predictor in the config.

## Config-based run (no long CLI)
- Edit `config/config.json` to set paths and hyperparameters.
- Run with:
```powershell
python scripts/run_config.py
```
Set `"mode": "train"` to train, or `"mode": "inference"` to run the webcam/video overlay with the configured paths.

### Capture via config
Set capture options in `config/config.json` (capture_out_dir, capture_metadata, capture_source, capture_every, capture_sleep) and run:
```powershell
python scripts/run_capture_config.py
```

### Balance positives with augmentation
To balance talking/not-talking counts by duplicating talking frames with color jitter + flips:
```powershell
python scripts/augment_balance.py
```
Uses metadata path from `config/config.json` (or override with `--metadata`). Augmented images are saved alongside originals and appended to the metadata CSV until positives match negatives.
Flags: `--max_factor` caps how many augmented positives you create (default 2.0x the original positives).

## Troubleshooting
- “No usable samples extracted”: paths in CSV invalid, images unreadable, or no face detected; recapture with good lighting/front-facing face.
- Face not detected: ensure `models/deploy.prototxt` and `models/res10_300x300_ssd_iter_140000.caffemodel` exist; the DNN face detector will be used before dlib/Haar.
- Use smaller `--every` during capture to save more frames; ensure the face is within frame at reasonable resolution.
