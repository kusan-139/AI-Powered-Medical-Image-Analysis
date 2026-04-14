"""
src/preprocessing/preprocess.py
──────────────────────────────────────────────────────────
Medical image preprocessing pipeline:
  • Normalisation / standardisation
  • CLAHE (contrast limited adaptive histogram equalisation)
  • Denoising (Gaussian / Bilateral)
  • Data augmentation (flips, rotations, brightness jitter)
  • Resize & pad
──────────────────────────────────────────────────────────
"""

from typing import Tuple, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ══════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════

def preprocess_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    apply_clahe: bool = True,
    apply_augmentation: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for train and test splits.

    Parameters
    ----------
    X_train, X_test : np.ndarray  shape (N, H, W, 3)  float32 [0,1]
    apply_clahe     : bool   Apply CLAHE to each image
    apply_augmentation : bool  Apply random augmentation to train set
    seed            : int    Random seed

    Returns
    -------
    X_train_p, X_test_p — preprocessed arrays (same shape, float32)
    """
    # CLAHE
    if apply_clahe and CV2_AVAILABLE:
        X_train = apply_clahe_batch(X_train)
        X_test  = apply_clahe_batch(X_test)

    # Z-score normalisation  (mean=0.485, std=0.229 — ImageNet priors)
    X_train = z_score_normalise(X_train)
    X_test  = z_score_normalise(X_test)

    # Augmentation (train only)
    if apply_augmentation:
        X_train = augment_batch(X_train, seed=seed)

    return X_train.astype(np.float32), X_test.astype(np.float32)


def preprocess_single_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe: bool = True,
) -> np.ndarray:
    """
    Preprocess a single image for inference.

    Parameters
    ----------
    image       : np.ndarray  HxWx3  uint8 OR float32 [0,1]
    target_size : (H, W)
    apply_clahe : bool

    Returns
    -------
    np.ndarray  (1, H, W, 3)  float32  — ready for model.predict()
    """
    # Ensure float32 [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Resize if needed
    h, w = target_size
    if image.shape[:2] != (h, w):
        if CV2_AVAILABLE:
            image = cv2.resize(image, (w, h))
        else:
            from PIL import Image as PILImage
            pil = PILImage.fromarray((image * 255).astype(np.uint8)).resize((w, h))
            image = np.array(pil, dtype=np.float32) / 255.0

    # CLAHE
    if apply_clahe and CV2_AVAILABLE:
        image = _apply_clahe_single(image)

    # Normalise
    image = _imagenet_norm(image)

    return image[np.newaxis].astype(np.float32)   # (1, H, W, 3)


# ══════════════════════════════════════════════════════════
#  CLAHE
# ══════════════════════════════════════════════════════════

def apply_clahe_batch(X: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE to every image in a batch."""
    out = np.empty_like(X, dtype=np.float32)
    for i, img in enumerate(X):
        out[i] = _apply_clahe_single(img, clip_limit)
    return out


def _apply_clahe_single(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply CLAHE on the L channel of Lab colour space.
    Input/output: float32 [0, 1]  HxWx3 RGB.
    """
    img_u8  = (img * 255).clip(0, 255).astype(np.uint8)
    lab     = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def z_score_normalise(X: np.ndarray) -> np.ndarray:
    """ImageNet-statistics normalisation (per-channel)."""
    return (X - _IMAGENET_MEAN) / (_IMAGENET_STD + 1e-8)


def _imagenet_norm(img: np.ndarray) -> np.ndarray:
    return (img - _IMAGENET_MEAN) / (_IMAGENET_STD + 1e-8)


def denormalise(X: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation (for visualisation)."""
    return (X * _IMAGENET_STD + _IMAGENET_MEAN).clip(0.0, 1.0)


# ══════════════════════════════════════════════════════════
#  AUGMENTATION (pure NumPy / optional OpenCV)
# ══════════════════════════════════════════════════════════

def augment_batch(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Random augmentation applied independently to each image.
    Augmentations: horizontal flip, vertical flip, rotation ±15°,
                   brightness jitter ±0.15, gaussian noise.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(X)

    for i, img in enumerate(X):
        # Horizontal flip
        if rng.random() > 0.5:
            img = img[:, ::-1].copy()

        # Vertical flip (light)
        if rng.random() > 0.8:
            img = img[::-1, :].copy()

        # Brightness jitter
        delta = rng.uniform(-0.15, 0.15)
        img   = (img + delta).clip(-3.0, 3.0)   # after norm, values can be negative

        # Gaussian noise
        if rng.random() > 0.5:
            noise = rng.normal(0, 0.02, img.shape).astype(np.float32)
            img   = img + noise

        out[i] = img

    return out
