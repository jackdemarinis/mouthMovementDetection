"""
Download pre-trained models for facial landmark detection.

This script downloads the required model files for dlib if needed.
For MediaPipe, models are included in the package.
"""

import os
import urllib.request
from pathlib import Path


def download_dlib_shape_predictor():
    """
    Download dlib's 68-point facial landmark predictor.
    """
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "shape_predictor_68_face_landmarks.dat"

    if model_path.exists():
        print(f"Model already exists: {model_path}")
        return

    print("Downloading dlib shape predictor...")
    print("This may take a few minutes...")

    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = model_dir / "shape_predictor_68_face_landmarks.dat.bz2"

    try:
        urllib.request.urlretrieve(url, compressed_path)
        print(f"Downloaded to: {compressed_path}")

        # Decompress
        import bz2
        print("Decompressing...")

        with bz2.open(compressed_path, 'rb') as source:
            with open(model_path, 'wb') as dest:
                dest.write(source.read())

        print(f"Model saved to: {model_path}")

        # Clean up compressed file
        compressed_path.unlink()
        print("Cleaned up compressed file")

    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nAlternative: Download manually from:")
        print(url)
        print(f"Extract and place in: {model_path}")


def main():
    print("="*60)
    print("MODEL DOWNLOAD UTILITY")
    print("="*60)
    print("\nNote: MediaPipe models are included in the package.")
    print("This script is only needed if you want to use dlib.\n")

    response = input("Download dlib shape predictor? (y/n): ")

    if response.lower() == 'y':
        download_dlib_shape_predictor()
    else:
        print("Skipping download.")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
