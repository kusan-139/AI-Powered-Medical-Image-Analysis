"""
src/inference/predictor.py
──────────────────────────────────────────────────────────
Inference pipeline:
  • Load a saved model (or build a mock for demo)
  • Preprocess a single image
  • Run prediction
  • Return structured result with class, confidence, Grad-CAM
──────────────────────────────────────────────────────────
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("Predictor")

CLASS_MAPS = {
    "pneumonia": {0: "Normal", 1: "Pneumonia"},
    "skin"     : {0: "Melanoma", 1: "Nevus", 2: "Basal Cell Carcinoma",
                  3: "Actinic Keratosis", 4: "Benign Keratosis",
                  5: "Dermatofibroma", 6: "Vascular Lesion"},
    "brain"    : {0: "No Tumour", 1: "Tumour"},
}


class Predictor:
    """
    Load a trained model and run inference on a single medical image.

    Usage:
        predictor = Predictor(task="pneumonia")
        result = predictor.predict("path/to/xray.jpg")
    """

    def __init__(self, task: str = "pneumonia"):
        if task not in CLASS_MAPS:
            raise ValueError(f"Unknown task: {task}")
        self.task      = task
        self.class_map = CLASS_MAPS[task]
        self.model     = self._load_model()

    # ──────────────────────────────────────────────────────
    def predict(
        self,
        image_path: str,
        generate_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image file.

        Parameters
        ----------
        image_path       : Path to JPEG / PNG image
        generate_gradcam : Save a Grad-CAM heatmap alongside result

        Returns
        -------
        dict with keys: class, confidence, class_id, gradcam_path
        """
        # Load + preprocess image
        image = self._load_image(image_path)
        if image is None:
            return {"error": f"Could not load image: {image_path}"}

        from preprocessing.preprocess import preprocess_single_image
        tensor = preprocess_single_image(image)   # (1, 224, 224, 3)

        # Predict
        raw = self.model.predict(tensor)

        # Parse output
        num_classes = len(self.class_map)
        if num_classes == 2:
            prob = float(raw[0, 1]) if raw.shape[1] > 1 else float(raw[0, 0])
            cls  = 1 if prob >= 0.5 else 0
            conf = prob if cls == 1 else 1 - prob
        else:
            cls  = int(raw[0].argmax())
            conf = float(raw[0, cls])

        result = {
            "class"     : self.class_map[cls],
            "class_id"  : cls,
            "confidence": round(conf, 4),
            "task"      : self.task,
        }

        # Grad-CAM
        if generate_gradcam:
            from explainability.gradcam import run_gradcam_demo
            gc_path = f"outputs/gradcam/{Path(image_path).stem}_gradcam.png"
            run_gradcam_demo(self.model, tensor, class_idx=cls, save_path=gc_path)
            result["gradcam_path"] = gc_path

        logger.info(f"Prediction: {result['class']} ({result['confidence']:.2%})")
        return result

    # ──────────────────────────────────────────────────────
    def _load_model(self):
        model_path = Path("models/saved") / f"{self.task}_model.keras"

        if model_path.exists():
            try:
                import tensorflow as tf
                logger.info(f"Loading saved model from {model_path}")
                return tf.keras.models.load_model(str(model_path))
            except Exception as e:
                logger.warning(f"Could not load saved model: {e} — using MockModel")

        logger.info("No saved model found — using MockModel for demo")
        from models.cnn_classifier import MockModel
        return MockModel(num_classes=len(self.class_map))

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            import cv2
            img = cv2.imread(path)
            if img is None:
                raise ValueError("cv2 returned None")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        except Exception:
            pass
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            return np.array(img, dtype=np.float32) / 255.0
        except Exception:
            return None
