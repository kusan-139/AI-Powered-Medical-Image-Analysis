"""
src/models/unet_segmentation.py
──────────────────────────────────────────────────────────
U-Net architecture for medical image segmentation.
Used for: Brain Tumour Segmentation (BraTS dataset)

Architecture:
  Encoder (4 blocks) → Bottleneck → Decoder (4 blocks)
  Skip connections connect encoder ↔ decoder at each level.
  Output: pixel-wise sigmoid mask.
──────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Tuple, Optional, Any


_TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except ImportError:
    pass


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 1,
    base_filters: int = 64,
) -> Any:
    """
    Build UNet model for medical image segmentation.

    Parameters
    ----------
    input_shape  : (H, W, C)   — grayscale MRI → C=1
    num_classes  : 1 for binary tumour mask, >1 for multi-class
    base_filters : Number of filters in first conv block (doubles per level)

    Returns
    -------
    Keras Model or MockSegmentationModel
    """
    if not _TF_AVAILABLE:
        print("  [INFO] TensorFlow not found — using MockSegmentationModel.")
        return MockSegmentationModel(input_shape=input_shape, num_classes=num_classes)

    return _build_keras_unet(input_shape, num_classes, base_filters)


def _conv_block(x, filters, name_prefix):
    x = keras.layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_c1")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_c2")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


def _build_keras_unet(input_shape, num_classes, F):
    inputs = keras.Input(shape=input_shape, name="mri_input")

    # ── Encoder ──────────────────────────────────────────
    e1 = _conv_block(inputs, F,     "enc1");   p1 = keras.layers.MaxPooling2D()(e1)
    e2 = _conv_block(p1,    F*2,   "enc2");   p2 = keras.layers.MaxPooling2D()(e2)
    e3 = _conv_block(p2,    F*4,   "enc3");   p3 = keras.layers.MaxPooling2D()(e3)
    e4 = _conv_block(p3,    F*8,   "enc4");   p4 = keras.layers.MaxPooling2D()(e4)

    # ── Bottleneck ────────────────────────────────────────
    b  = _conv_block(p4, F*16, "bottleneck")

    # ── Decoder ──────────────────────────────────────────
    d4 = keras.layers.Conv2DTranspose(F*8,  2, strides=2, padding="same")(b)
    d4 = keras.layers.Concatenate()([d4, e4])
    d4 = _conv_block(d4, F*8,  "dec4")

    d3 = keras.layers.Conv2DTranspose(F*4,  2, strides=2, padding="same")(d4)
    d3 = keras.layers.Concatenate()([d3, e3])
    d3 = _conv_block(d3, F*4,  "dec3")

    d2 = keras.layers.Conv2DTranspose(F*2,  2, strides=2, padding="same")(d3)
    d2 = keras.layers.Concatenate()([d2, e2])
    d2 = _conv_block(d2, F*2,  "dec2")

    d1 = keras.layers.Conv2DTranspose(F,    2, strides=2, padding="same")(d2)
    d1 = keras.layers.Concatenate()([d1, e1])
    d1 = _conv_block(d1, F,    "dec1")

    # ── Output ────────────────────────────────────────────
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = keras.layers.Conv2D(
        num_classes, 1, activation=activation, name="segmentation_mask"
    )(d1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="UNet_MedSeg")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=_dice_loss,
        metrics=["accuracy", _dice_coeff],
    )
    return model


def _dice_coeff(y_true, y_pred, smooth=1.0):
    """Dice similarity coefficient — standard in medical segmentation."""
    import tensorflow as tf
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def _dice_loss(y_true, y_pred):
    return 1.0 - _dice_coeff(y_true, y_pred)


# ══════════════════════════════════════════════════════════
#  MOCK SEGMENTATION MODEL
# ══════════════════════════════════════════════════════════

class MockSegmentationModel:
    def __init__(
        self,
        input_shape=(256, 256, 1),
        num_classes=1,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self._rng = np.random.default_rng(42)

    def count_params(self):
        return 31_000_000

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        N, H, W = X.shape[:3]
        # Return a random binary-ish mask
        masks = self._rng.random((N, H, W, self.num_classes)).astype(np.float32)
        return (masks > 0.7).astype(np.float32)

    def summary(self, print_fn=print):
        print_fn(f"MockUNet(input={self.input_shape}, classes={self.num_classes})")
        print_fn("  Total params: 31,000,000 (mock)")

    def fit(self, *args, **kwargs):
        return type("History", (), {"history": {"loss": [0.4, 0.3], "accuracy": [0.8, 0.85]}})()
