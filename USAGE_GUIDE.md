# Mouth Movement Detection - Usage Guide

Complete guide to using the mouth movement detection system.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Collection](#data-collection)
4. [Data Labeling](#data-labeling)
5. [Training](#training)
6. [Inference](#inference)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jackdemarinis/Mouth-Movement-Detection.git
cd Mouth-Movement-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with `dlib`, use MediaPipe instead (recommended).

### 4. Verify Installation

```bash
python -c "import torch; import cv2; import mediapipe; print('All dependencies installed!')"
```

---

## Quick Start

### Step 1: Collect Data (5-10 minutes)

```bash
python src/data_processing/collect_data.py --output data/raw --duration 300
```

**Controls:**
- `SPACE`: Start/Stop recording
- `s`: Save session
- `q`: Quit

**Tips:**
- Record yourself speaking for 2-3 minutes
- Record yourself silent for 2-3 minutes
- Ensure good lighting
- Look at the camera

### Step 2: Label Data

**Option A: Auto-labeling** (faster, less accurate)
```bash
python src/data_processing/label_data.py --session data/raw/[SESSION_NAME] --mode auto
```

**Option B: Manual labeling** (slower, more accurate)
```bash
python src/data_processing/label_data.py --session data/raw/[SESSION_NAME] --mode manual
```

Manual labeling controls:
- `1` or `→`: Label as MOVING
- `0` or `←`: Label as NOT MOVING
- `SPACE`: Toggle label
- `s`: Save
- `q`: Quit

### Step 3: Train Model

```bash
python src/models/train.py --config config/train_config.yaml
```

This will:
- Load your collected data
- Split into train/val/test sets (70%/15%/15%)
- Train the neural network
- Save the best model to `models/checkpoints/best_model.pth`
- Print evaluation metrics

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir logs
```

### Step 4: Run Inference

```bash
python src/inference.py --model models/checkpoints/best_model.pth
```

**Controls:**
- `q`: Quit
- `s`: Save screenshot
- `r`: Reset temporal history
- `l`: Toggle landmarks
- `f`: Toggle features

---

## Data Collection

### Collecting Quality Data

1. **Speaking Data (MOVING class):**
   - Speak naturally (read a book, have a conversation)
   - Vary your speech rate (fast, slow, normal)
   - Include different volumes (whisper, normal, loud)
   - Duration: At least 2-3 minutes

2. **Silent Data (NOT MOVING class):**
   - Keep mouth closed
   - Try different expressions (smile, neutral, frown)
   - Include natural movements (blinking, head turning)
   - Duration: At least 2-3 minutes

3. **Environment:**
   - Good lighting (avoid backlit situations)
   - Stable camera position
   - Clear background
   - Face camera directly

### Multiple Sessions

Collect multiple sessions for better model generalization:

```bash
python src/data_processing/collect_data.py --session session1 --duration 300
python src/data_processing/collect_data.py --session session2 --duration 300
python src/data_processing/collect_data.py --session session3 --duration 300
```

### Data Augmentation

Collect data under varied conditions:
- Different lighting (bright, dim, natural, artificial)
- Different head angles (±15°, ±30°)
- Different distances from camera
- With/without glasses

---

## Data Labeling

### Auto-labeling

Auto-labeling uses temporal feature variance to classify frames:

```bash
python src/data_processing/label_data.py \
    --session data/raw/[SESSION_NAME] \
    --mode auto \
    --threshold 60
```

**Threshold parameter:**
- Higher (70-80): More conservative (fewer false positives)
- Lower (40-50): More sensitive (more detections)
- Default (60): Balanced

### Manual Labeling

For better accuracy, manually label a subset of data:

```bash
python src/data_processing/label_data.py \
    --session data/raw/[SESSION_NAME] \
    --mode manual
```

**Best practices:**
- Label at least 500-1000 frames manually
- Focus on ambiguous cases (slight movements, expressions)
- Take breaks to maintain consistency

### Hybrid Approach

1. Auto-label most data
2. Manually review and correct difficult cases
3. Use manual labels as validation set

---

## Training

### Basic Training

```bash
python src/models/train.py --config config/train_config.yaml
```

### Configuration

Edit `config/train_config.yaml` to customize training:

```yaml
model:
  input_size: 25          # Feature dimension
  hidden_size: 12         # Hidden layer neurons
  dropout: 0.2            # Dropout rate

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  early_stopping_patience: 10
```

### Hyperparameter Tuning

Try different configurations:

**For better accuracy:**
- Increase hidden_size (16, 20)
- Decrease dropout (0.1, 0.15)
- Increase epochs (100, 150)

**For faster inference:**
- Decrease hidden_size (8, 10)
- Keep dropout moderate (0.2)

**For better generalization:**
- Increase dropout (0.3, 0.4)
- Use more training data
- Enable data augmentation

### Monitoring Training

View training progress in TensorBoard:

```bash
tensorboard --logdir logs
```

Open browser to `http://localhost:6006`

**Metrics to watch:**
- Training loss (should decrease)
- Validation loss (should decrease, shouldn't diverge from training)
- F1-score (should increase, target >0.85)
- Accuracy (target >90%)

### Resume Training

Resume from a checkpoint:

```bash
python src/models/train.py --config config/train_config.yaml --resume models/checkpoints/model_epoch_20.pth
```

---

## Inference

### Real-Time Webcam

```bash
python src/inference.py --model models/checkpoints/best_model.pth
```

### Custom Threshold

Adjust classification threshold:

```bash
python src/inference.py --model models/checkpoints/best_model.pth --threshold 0.6
```

- Higher threshold (0.6-0.8): Fewer false positives, may miss subtle movements
- Lower threshold (0.3-0.4): More sensitive, more false positives

### GPU Inference

If you have CUDA available:

```bash
python src/inference.py --model models/checkpoints/best_model.pth --device cuda
```

### Performance Metrics

The inference script displays:
- **FPS**: Frames per second (target: ≥30)
- **Inference Time**: Milliseconds per frame (target: <10ms)
- **Confidence**: Model probability (0-100%)

---

## Troubleshooting

### Installation Issues

**Problem:** `dlib` installation fails

**Solution:** Use MediaPipe instead (already set as default)

**Problem:** `torch` installation fails

**Solution:**
```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Data Collection Issues

**Problem:** No face detected

**Solutions:**
- Ensure good lighting
- Move closer to camera
- Clean camera lens
- Check camera is working: `cv2.VideoCapture(0)`

**Problem:** Landmarks jumping around

**Solutions:**
- Stay still during recording
- Ensure stable lighting
- Reduce background motion

### Training Issues

**Problem:** Model not learning (accuracy ~50%)

**Solutions:**
- Check data is properly labeled
- Ensure balanced classes
- Increase model capacity (hidden_size)
- Collect more diverse data

**Problem:** Overfitting (train accuracy high, val accuracy low)

**Solutions:**
- Increase dropout (0.3-0.4)
- Reduce model size (hidden_size)
- Collect more training data
- Enable data augmentation

**Problem:** Training very slow

**Solutions:**
- Reduce batch size
- Use GPU if available
- Reduce dataset size for testing

### Inference Issues

**Problem:** Low FPS (<30)

**Solutions:**
- Use CPU inference (may be faster for small models)
- Reduce image resolution
- Skip frames (process every 2nd or 3rd frame)

**Problem:** Poor real-time accuracy

**Solutions:**
- Collect more diverse training data
- Adjust classification threshold
- Retrain with real-world examples
- Check lighting conditions

**Problem:** Many false positives (classifies smiling as speaking)

**Solutions:**
- Include facial expressions in NOT MOVING training data
- Increase classification threshold (0.6-0.7)
- Add temporal smoothing

---

## Advanced Usage

### Using Jupyter Notebooks

```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

Interactive environment for:
- Data exploration
- Feature visualization
- Model experimentation
- Result analysis

### Custom Features

Add custom features in `src/features/mouth_features.py`:

```python
def compute_custom_feature(self, landmarks):
    # Your feature computation
    return feature_value
```

### Custom Model Architecture

Modify `src/models/network.py` to experiment with:
- Different activation functions
- Multiple hidden layers
- Batch normalization
- Different regularization techniques

---

## Performance Benchmarks

Expected performance on standard hardware:

| Metric | Target | Typical |
|--------|--------|---------|
| Accuracy | ≥90% | 92-95% |
| F1-Score | ≥0.85 | 0.88-0.92 |
| Inference Time | <10ms | 5-8ms |
| FPS | ≥30 | 35-50 |

Hardware tested:
- CPU: Intel i5/i7 or AMD Ryzen 5/7
- RAM: 8GB+
- No GPU required

---

## Next Steps

1. **Improve accuracy:**
   - Collect more diverse data
   - Try deeper network architecture
   - Experiment with different features

2. **Deploy application:**
   - Integrate with video conferencing
   - Build speaker identification system
   - Create accessibility tools

3. **Optimize performance:**
   - Model quantization
   - ONNX export
   - Mobile deployment

---

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/jackdemarinis/Mouth-Movement-Detection/issues
- Documentation: See README.md
- Examples: See notebooks/

---

**Good luck with your mouth movement detection project!** 🎯
