# Quick Start Guide - Mouth Movement Detection

Get started with mouth movement detection in 4 simple steps!

## Prerequisites

- Python 3.8+
- Webcam
- 15-20 minutes

## Step 1: Install Dependencies (2 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

## Step 2: Collect Training Data (5-10 minutes)

```bash
python src/data_processing/collect_data.py --output data/raw --duration 300
```

### What to do:
1. Press `SPACE` to start recording
2. **Speak naturally for 2-3 minutes** (read aloud, count, talk)
3. Press `SPACE` to stop
4. **Stay silent for 2-3 minutes** (keep mouth closed)
5. Press `s` to save
6. Press `q` to quit

### Tips:
- Face the camera
- Good lighting is important
- Try different head angles

## Step 3: Label and Train (5 minutes)

```bash
# Auto-label your data
python src/data_processing/label_data.py --session data/raw/[YOUR_SESSION_NAME] --mode auto

# Train the model
python src/models/train.py --config config/train_config.yaml
```

**Replace `[YOUR_SESSION_NAME]`** with the timestamp folder created in `data/raw/`

Training will show progress and save the best model automatically.

## Step 4: Test It Live! (30 seconds)

```bash
python src/inference.py --model models/checkpoints/best_model.pth
```

You should see:
- Your webcam feed
- Green box when mouth is moving (speaking)
- Red box when mouth is still
- FPS and confidence scores

Press `q` to quit.

## Expected Results

With 5-10 minutes of good training data:
- **Accuracy:** 85-95%
- **FPS:** 30-50
- **Inference Time:** 5-10ms

## Troubleshooting

### No face detected?
- Move closer to camera
- Improve lighting
- Ensure camera is working

### Poor accuracy?
- Collect more diverse data
- Try manual labeling instead of auto
- Include varied conditions (speaking styles, expressions)

### Slow performance?
- Normal on older computers
- Model is optimized for CPU
- Should still achieve 30+ FPS

## Next Steps

- Read [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed information
- Try the Jupyter notebook: `notebooks/01_getting_started.ipynb`
- Experiment with different configurations in `config/train_config.yaml`
- Collect more data for better accuracy

## Full Workflow Summary

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Collect data
python src/data_processing/collect_data.py --output data/raw --duration 300

# 3. Label data
python src/data_processing/label_data.py --session data/raw/[SESSION] --mode auto

# 4. Train model
python src/models/train.py --config config/train_config.yaml

# 5. Run inference
python src/inference.py --model models/checkpoints/best_model.pth
```

**That's it! You now have a working mouth movement detector!** 🎉

---

**Questions?** Check the [README.md](README.md) or [USAGE_GUIDE.md](USAGE_GUIDE.md)
