"""
Verify Installation Script

Checks that all dependencies are installed correctly and the system is ready to use.
"""

import sys
import importlib


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT installed")
        return False


def check_opencv():
    """Check OpenCV and camera access."""
    try:
        import cv2
        print(f"✓ OpenCV version {cv2.__version__}")

        # Try to open camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  ✓ Camera access OK")
            cap.release()
        else:
            print("  ⚠ Warning: Could not access camera")

        return True
    except:
        print("✗ OpenCV NOT working")
        return False


def check_torch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"✓ PyTorch version {torch.__version__}")

        device = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"  → Device: {device}")

        if device == "CUDA":
            print(f"  → GPU: {torch.cuda.get_device_name(0)}")

        return True
    except:
        print("✗ PyTorch NOT installed")
        return False


def check_mediapipe():
    """Check MediaPipe installation."""
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe installed")

        # Try to initialize face mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1
        )
        print("  ✓ Face mesh initialization OK")
        face_mesh.close()

        return True
    except Exception as e:
        print(f"✗ MediaPipe issue: {e}")
        return False


def check_project_structure():
    """Check if project structure is correct."""
    import os
    from pathlib import Path

    required_dirs = [
        'src',
        'src/features',
        'src/models',
        'src/data_processing',
        'src/utils',
        'data',
        'data/raw',
        'data/processed',
        'models',
        'models/checkpoints',
        'config',
        'notebooks'
    ]

    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ NOT found")
            all_good = False

    return all_good


def main():
    print("="*60)
    print("MOUTH MOVEMENT DETECTION - INSTALLATION VERIFICATION")
    print("="*60)
    print()

    print("Checking Python version...")
    print(f"Python {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠ Warning: Python 3.8+ recommended")
    print()

    print("Checking core dependencies...")
    checks = []

    # Core packages
    checks.append(check_torch())
    checks.append(check_opencv())
    checks.append(check_mediapipe())
    checks.append(check_import("numpy"))
    checks.append(check_import("scipy"))
    checks.append(check_import("sklearn", "scikit-learn"))

    print()
    print("Checking visualization libraries...")
    checks.append(check_import("matplotlib"))
    checks.append(check_import("seaborn"))

    print()
    print("Checking utility libraries...")
    checks.append(check_import("yaml", "pyyaml"))
    checks.append(check_import("tqdm"))
    checks.append(check_import("PIL", "Pillow"))

    print()
    print("Checking Jupyter...")
    checks.append(check_import("jupyter"))

    print()
    print("Checking project structure...")
    structure_ok = check_project_structure()

    print()
    print("="*60)

    if all(checks) and structure_ok:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("Your system is ready to use!")
        print()
        print("Next steps:")
        print("1. Read QUICKSTART.md for a quick tutorial")
        print("2. Run: python src/data_processing/collect_data.py")
        print("   to start collecting training data")
    else:
        print("⚠ SOME CHECKS FAILED")
        print()
        print("To fix:")
        print("1. Activate your virtual environment")
        print("2. Run: pip install -r requirements.txt")
        print("3. Run this script again")

    print("="*60)
    print()


if __name__ == "__main__":
    main()
