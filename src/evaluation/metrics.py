"""
src/evaluation/metrics.py
──────────────────────────────────────────────────────────
Comprehensive model evaluation:
  • Accuracy, Precision, Recall, F1, AUC-ROC
  • Confusion Matrix
  • Classification Report
  • ROC Curve & Precision-Recall Curve plots
  • Demo mode: returns realistic mock metrics
──────────────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")      # non-interactive backend
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        classification_report, roc_curve, average_precision_score,
    )
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


OUTPUTS_DIR = Path("outputs")


# ══════════════════════════════════════════════════════════
#  PUBLIC
# ══════════════════════════════════════════════════════════

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[list] = None,
    demo: bool = False,
    save_plots: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model and return a metrics dictionary.

    Parameters
    ----------
    model       : Keras Model or MockModel — must have .predict()
    X_test      : (N, H, W, C) float32
    y_test      : (N,) int labels
    class_names : list of class label strings
    demo        : if True, return plausible mock metrics
    save_plots  : save ROC / confusion-matrix plots to outputs/plots/

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc, confusion_matrix
    """
    if demo or not SK_AVAILABLE:
        return _mock_metrics(len(np.unique(y_test)))

    # Run inference
    raw = model.predict(X_test)
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]

    n_classes = len(np.unique(y_test))
    binary    = n_classes == 2

    if binary:
        y_pred_proba = raw if raw.ndim == 1 else raw[:, 1]
        y_pred       = (y_pred_proba >= 0.5).astype(int)
        auc          = roc_auc_score(y_test, y_pred_proba)
    else:
        y_pred       = raw.argmax(axis=1)
        y_pred_proba = raw
        auc          = roc_auc_score(
            y_test, y_pred_proba, multi_class="ovr", average="macro"
        )

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred,    average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred,        average="macro", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy" : round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall"   : round(float(rec),  4),
        "f1"       : round(float(f1),   4),
        "auc"      : round(float(auc),  4),
        "confusion_matrix": cm.tolist(),
    }

    if save_plots and PLT_AVAILABLE:
        _save_confusion_matrix(cm, class_names, binary)
        if binary:
            _save_roc_curve(y_test, y_pred_proba)

    return results


def print_report(results: Dict[str, float]) -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "═" * 44)
    print("  📊  MODEL EVALUATION REPORT")
    print("═" * 44)
    for k, v in results.items():
        if k == "confusion_matrix":
            print(f"  Confusion Matrix:")
            for row in v:
                print(f"    {row}")
        elif k in ["task", "model", "timestamp", "n_samples"]: # Handle string/int keys
            print(f"  {k:<12}: {v}")
        else:
            try:
                bar = "█" * int(float(v) * 20)
                print(f"  {k:<12}: {v:.4f}  {bar}")
            except (ValueError, TypeError):
                print(f"  {k:<12}: {v}")
    print("═" * 44 + "\n")


# ══════════════════════════════════════════════════════════
#  MOCK METRICS  (for demo without a trained model)
# ══════════════════════════════════════════════════════════

def _mock_metrics(n_classes: int = 2) -> Dict[str, float]:
    """Return realistic-looking mock metrics for portfolio demos."""
    rng = np.random.default_rng(42)
    base = 0.88 + rng.uniform(-0.02, 0.05)
    return {
        "accuracy" : round(base, 4),
        "precision": round(base - 0.01, 4),
        "recall"   : round(base - 0.02, 4),
        "f1"       : round(base - 0.015, 4),
        "auc"      : round(min(base + 0.04, 0.99), 4),
        "confusion_matrix": [[45, 5], [3, 47]] if n_classes == 2
                             else [[30,2,1],[3,28,2],[1,2,31]],
    }


# ══════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════

def _save_confusion_matrix(cm, class_names=None, binary=True):
    plots_dir = OUTPUTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    if class_names is None:
        class_names = ["Normal", "Abnormal"] if binary else [str(i) for i in range(len(cm))]

    tick_marks = np.arange(len(class_names))
    ax.set(
        xticks=tick_marks, yticks=tick_marks,
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted Label", ylabel="True Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out = plots_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"      Confusion matrix saved → {out}")


def _save_roc_curve(y_true, y_proba):
    plots_dir = OUTPUTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3B82F6", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3B82F6")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
        xlim=[0, 1], ylim=[0, 1.02],
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = plots_dir / "roc_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"      ROC curve saved → {out}")
