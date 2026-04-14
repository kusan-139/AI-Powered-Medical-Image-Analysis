"""
src/data/data_loader.py
──────────────────────────────────────────────────────────
Handles:
  1. Synthetic dataset generation (for demo / unit-tests)
  2. Real dataset loaders for:
       • Chest X-Ray14  (Pneumonia / Normal)
       • HAM10000       (Skin Lesion Classification)
       • BraTS          (Brain Tumour Segmentation)
──────────────────────────────────────────────────────────
"""

import os
import glob
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ══════════════════════════════════════════════════════════
#  SYNTHETIC DATASET — works with zero external data
# ══════════════════════════════════════════════════════════

def generate_synthetic_dataset(
    n_samples: int = 300,
    img_size: Tuple[int, int] = (224, 224),
    n_classes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic medical-image dataset for demo purposes.

    Returns
    -------
    X_train, X_test, y_train, y_test  — all numpy arrays
        X shape : (N, H, W, 3)  float32 [0, 1]
        y shape : (N,)           int
    """
    rng = np.random.default_rng(seed)
    H, W = img_size

    images = []
    labels = []

    for cls in range(n_classes):
        n = n_samples // n_classes
        for _ in range(n):
            # Base image: uniform noise (mimics X-ray texture)
            img = rng.random((H, W, 3), dtype=np.float32) * 0.3

            # Add class-specific pattern
            if cls == 1:
                # Simulate consolidation / infiltrate
                cx, cy = rng.integers(60, H - 60), rng.integers(60, W - 60)
                r  = rng.integers(20, 60)
                yy, xx = np.ogrid[:H, :W]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
                img[mask] += rng.random() * 0.5

            img = np.clip(img, 0.0, 1.0)
            images.append(img)
            labels.append(cls)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════
#  REAL DATASET LOADER — Chest X-Ray14
# ══════════════════════════════════════════════════════════

def load_chest_xray(
    data_dir: str = "data/raw/chest_xray",
    img_size: Tuple[int, int] = (224, 224),
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Chest X-Ray 2-class dataset (NORMAL / PNEUMONIA).

    Dataset download (free, no login):
      https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

    Expected folder structure:
      data/raw/chest_xray/
          train/NORMAL/
          train/PNEUMONIA/
          test/NORMAL/
          test/PNEUMONIA/

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    class_map = {"NORMAL": 0, "PNEUMONIA": 1}
    splits = {"train": [], "test": []}

    for split in splits:
        for cls_name, cls_idx in class_map.items():
            folder = Path(data_dir) / split / cls_name
            if not folder.exists():
                continue
            files = list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            if max_samples:
                files = files[: max_samples // 4]
            for fpath in files:
                img = _load_image(str(fpath), img_size)
                if img is not None:
                    splits[split].append((img, cls_idx))

    def _unzip(pairs):
        if not pairs:
            return np.empty((0, *img_size, 3), dtype=np.float32), np.empty(0, dtype=np.int64)
        imgs, lbls = zip(*pairs)
        return np.array(imgs, dtype=np.float32), np.array(lbls, dtype=np.int64)

    X_train, y_train = _unzip(splits["train"])
    X_test,  y_test  = _unzip(splits["test"])

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════
#  REAL DATASET LOADER — HAM10000 Skin Lesion
# ══════════════════════════════════════════════════════════

def load_ham10000(
    data_dir: str = "data/raw/skin_lesion",
    img_size: Tuple[int, int] = (224, 224),
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load HAM10000 skin lesion dataset.

    Dataset download (free):
      https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

    Expected structure:
      data/raw/skin_lesion/
          HAM10000_images_part_1/  *.jpg
          HAM10000_images_part_2/  *.jpg
          HAM10000_metadata.csv

    Returns
    -------
    X_train, X_test, y_train, y_test
    (7 classes: mel, nv, bcc, akiec, bkl, df, vasc)
    """
    import pandas as pd

    meta_path = Path(data_dir) / "HAM10000_metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {meta_path}. "
            "Please download HAM10000 from Kaggle."
        )

    df = pd.read_csv(meta_path)
    label_map = {v: i for i, v in enumerate(df["dx"].unique())}
    df["label"] = df["dx"].map(label_map)

    if max_samples:
        df = df.sample(max_samples, random_state=42)

    images, labels = [], []
    img_dirs = [
        Path(data_dir) / "HAM10000_images_part_1",
        Path(data_dir) / "HAM10000_images_part_2",
    ]

    for _, row in df.iterrows():
        img_file = row["image_id"] + ".jpg"
        for d in img_dirs:
            p = d / img_file
            if p.exists():
                img = _load_image(str(p), img_size)
                if img is not None:
                    images.append(img)
                    labels.append(row["label"])
                break

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def _load_image(
    path: str,
    size: Tuple[int, int] = (224, 224),
) -> Optional[np.ndarray]:
    """Load, resize and normalise a single image to float32 [0,1]."""
    if CV2_AVAILABLE:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size[::-1])   # cv2 takes (W, H)
        return img.astype(np.float32) / 255.0

    if PIL_AVAILABLE:
        img = Image.open(path).convert("RGB").resize(size[::-1])
        return np.array(img, dtype=np.float32) / 255.0

    return None
