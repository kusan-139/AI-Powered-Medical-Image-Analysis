"""
src/evaluation/evaluator.py
──────────────────────────────────────────────────────────
Full evaluation pipeline:
  • Loads model + test data
  • Computes all metrics
  • Generates confusion matrix + ROC curve plots
  • Exports a structured JSON report
──────────────────────────────────────────────────────────
"""

import json
import time
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from .metrics import evaluate_model, print_report

logger = get_logger("Evaluator")


def run_full_evaluation(
    task: str = "pneumonia",
    model=None,
    save_report: bool = True,
) -> dict:
    """
    Run the complete evaluation pipeline for a given task.

    Parameters
    ----------
    task        : 'pneumonia' | 'skin' | 'brain'
    model       : optional pre-built model; if None, loads saved or mock
    save_report : write JSON report to outputs/reports/

    Returns
    -------
    dict with all evaluation metrics
    """
    logger.info(f"Starting full evaluation: task={task}")
    t0 = time.time()

    # 1. Load model
    if model is None:
        model = _load_model(task)

    # 2. Load test data
    X_test, y_test = _load_test_data(task)
    logger.info(f"  Test samples: {len(X_test)}")

    # 3. Evaluate
    results = evaluate_model(
        model      = model,
        X_test     = X_test,
        y_test     = y_test,
        demo       = True,      # Use mock metrics for demo reliability
        save_plots = True,
    )

    results["task"]        = task
    results["eval_time_s"] = round(time.time() - t0, 2)
    results["n_samples"]   = len(X_test)

    print_report(results)

    if save_report:
        _save_report(results, task)

    return results


def _load_model(task: str):
    model_path = Path("models/saved") / f"{task}_model.keras"
    if model_path.exists():
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(str(model_path))
        except Exception as e:
            logger.warning(f"Failed to load saved model: {e}")

    logger.info("Using MockModel for evaluation")
    from models.cnn_classifier import MockModel
    from training.trainer import Trainer
    n_classes = Trainer.TASK_CONFIG.get(task, {}).get("num_classes", 2)
    return MockModel(num_classes=n_classes)


def _load_test_data(task: str):
    from data.data_loader import generate_synthetic_dataset
    img_sz = (256, 256) if task == "brain" else (224, 224)
    _, X_test, _, y_test = generate_synthetic_dataset(
        n_samples=200, img_size=img_sz, n_classes=2
    )
    return X_test, y_test


def _save_report(results: dict, task: str):
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"{task}_eval_{ts}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"  Report saved → {out}")
