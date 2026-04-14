"""
src/training/trainer.py
──────────────────────────────────────────────────────────
Training orchestrator — handles:
  • Model selection & creation
  • Dataset loading (real or synthetic)
  • Training loop with callbacks
  • Checkpoint saving
  • Training history plots
──────────────────────────────────────────────────────────
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("Trainer")


class Trainer:
    """
    Unified trainer for all tasks:
      - pneumonia  : binary classification on Chest X-Ray14
      - skin       : 7-class HAM10000 skin lesion classification
      - brain      : segmentation on synthetic BraTS-style data
    """

    TASK_CONFIG = {
        "pneumonia": {
            "num_classes": 2,
            "class_names": ["Normal", "Pneumonia"],
            "backbone": "mobilenet_v2",
            "loader": "chest_xray",
        },
        "skin": {
            "num_classes": 7,
            "class_names": ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"],
            "backbone": "efficientnet_b0",
            "loader": "ham10000",
        },
        "brain": {
            "num_classes": 1,
            "class_names": ["Tumour"],
            "backbone": "unet",
            "loader": "synthetic",
        },
    }

    def __init__(
        self,
        task: str = "pneumonia",
        epochs: int = 10,
        batch_size: int = 32,
        config: Optional[Dict[str, Any]] = None,
    ):
        if task not in self.TASK_CONFIG:
            raise ValueError(f"Unknown task '{task}'. Choose: {list(self.TASK_CONFIG.keys())}")

        self.task       = task
        self.epochs     = epochs
        self.batch_size = batch_size
        self.config     = config or {}
        self.task_cfg   = self.TASK_CONFIG[task]

        self.model_dir  = Path("models/saved")
        self.ckpt_dir   = Path("models/checkpoints")
        self.plots_dir  = Path("outputs/plots")
        for d in [self.model_dir, self.ckpt_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────
    def run(self):
        """Main training entry point."""
        logger.info(f"Starting training: task={self.task}, epochs={self.epochs}")

        # 1. Load data
        X_train, X_test, y_train, y_test = self._load_data()
        logger.info(f"  Data: train={len(X_train)}, test={len(X_test)}")

        # 2. Preprocess
        from preprocessing.preprocess import preprocess_pipeline
        X_train, X_test = preprocess_pipeline(X_train, X_test)

        # 3. Build model
        model = self._build_model()

        # 4. Train
        history = self._train(model, X_train, y_train, X_test, y_test)

        # 5. Save
        self._save_model(model)
        self._save_history(history)
        self._plot_history(history)

        logger.info("✅  Training complete!")

    # ──────────────────────────────────────────────────────
    def _load_data(self):
        loader = self.task_cfg["loader"]

        if loader == "chest_xray":
            try:
                from data.data_loader import load_chest_xray
                return load_chest_xray()
            except FileNotFoundError:
                logger.warning("Chest X-Ray data not found — using synthetic data")

        elif loader == "ham10000":
            try:
                from data.data_loader import load_ham10000
                return load_ham10000()
            except FileNotFoundError:
                logger.warning("HAM10000 data not found — using synthetic data")

        # Fallback: synthetic
        from data.data_loader import generate_synthetic_dataset
        n_cls = self.task_cfg["num_classes"]
        img_sz = (256, 256) if self.task == "brain" else (224, 224)
        return generate_synthetic_dataset(
            n_samples=300, img_size=img_sz, n_classes=max(n_cls, 2)
        )

    def _build_model(self):
        backbone = self.task_cfg["backbone"]
        n_classes = self.task_cfg["num_classes"]

        if backbone == "unet":
            from models.unet_segmentation import build_unet
            return build_unet(input_shape=(256, 256, 3), num_classes=n_classes)
        else:
            from models.cnn_classifier import build_cnn_model
            return build_cnn_model(
                input_shape=(224, 224, 3),
                num_classes=n_classes,
                backbone=backbone,
            )

    def _train(self, model, X_train, y_train, X_val, y_val):
        logger.info("  Starting training loop …")
        t0 = time.time()

        history_obj = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

        elapsed = time.time() - t0
        logger.info(f"  Training finished in {elapsed:.1f}s")

        history = getattr(history_obj, "history", {}) or {}
        return history

    def _save_model(self, model):
        out = self.model_dir / f"{self.task}_model.json"
        meta = {
            "task"     : self.task,
            "backbone" : self.task_cfg["backbone"],
            "classes"  : self.task_cfg["class_names"],
            "framework": "keras" if hasattr(model, "save") else "mock",
        }
        out.write_text(json.dumps(meta, indent=2))

        if hasattr(model, "save"):
            try:
                model.save(str(self.model_dir / f"{self.task}_model.keras"))
                logger.info(f"  Model saved → {self.model_dir}")
            except Exception as e:
                logger.warning(f"  Could not save Keras model: {e}")

    def _save_history(self, history: dict):
        out = self.plots_dir / f"{self.task}_history.json"
        out.write_text(json.dumps(history, indent=2))

    def _plot_history(self, history: dict):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            metrics_to_plot = [k for k in history
                               if not k.startswith("val_")]
            if not metrics_to_plot:
                return

            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5))
            if len(metrics_to_plot) == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics_to_plot):
                ax.plot(history[metric], label="Train", lw=2, color="#3B82F6")
                val_key = f"val_{metric}"
                if val_key in history:
                    ax.plot(history[val_key], label="Val", lw=2, color="#F97316", linestyle="--")
                ax.set(title=metric.capitalize(), xlabel="Epoch", ylabel=metric)
                ax.legend()
                ax.grid(alpha=0.3)

            plt.suptitle(f"{self.task.capitalize()} — Training History", fontsize=14)
            plt.tight_layout()
            out = self.plots_dir / f"{self.task}_training_history.png"
            plt.savefig(out, dpi=150)
            plt.close()
            logger.info(f"  Training plot saved → {out}")

        except Exception as e:
            logger.warning(f"  Could not save training plot: {e}")
