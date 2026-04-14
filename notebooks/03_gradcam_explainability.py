"""
notebooks/03_gradcam_explainability.py
──────────────────────────────────────────────────────────
Demonstrations of Grad-CAM explainability:
  1. Generate synthetic Grad-CAM heatmaps
  2. Visualise overlay on chest X-rays
  3. Show how regions are highlighted per class
  4. Discussion of clinical relevance
──────────────────────────────────────────────────────────
Execute: python notebooks/03_gradcam_explainability.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cmap_module

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.data_loader import generate_synthetic_dataset
from models.cnn_classifier import build_cnn_model
from preprocessing.preprocess import preprocess_pipeline, denormalise


def section(t):
    print("\n" + "═"*56 + f"\n  {t}\n" + "═"*56)


# ════════════════════════════════════════════════════════
section("1 · Prepare Data & Model")
# ════════════════════════════════════════════════════════
X_tr, X_te, y_tr, y_te = generate_synthetic_dataset(
    n_samples=60, img_size=(224, 224), n_classes=2, seed=99
)
X_tr_p, X_te_p = preprocess_pipeline(X_tr, X_te)

model = build_cnn_model(input_shape=(224, 224, 3), num_classes=2)
print("  Model ready ✓")


# ════════════════════════════════════════════════════════
section("2 · Generate Grad-CAM Heatmaps")
# ════════════════════════════════════════════════════════
def synthetic_gradcam(img, n_blobs=3, seed=7):
    """Synthetic heatmap (mimics real Grad-CAM output)."""
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    hmap = np.zeros((H, W), dtype=np.float32)
    for _ in range(n_blobs):
        cx, cy = rng.integers(40, W-40), rng.integers(40, H-40)
        r   = rng.integers(20, 70)
        str_= rng.uniform(0.4, 1.0)
        y_, x_ = np.ogrid[:H, :W]
        blob = np.exp(-((x_-cx)**2+(y_-cy)**2)/(2*r**2)) * str_
        hmap = np.maximum(hmap, blob)
    hmap /= hmap.max() + 1e-8
    return hmap


# ════════════════════════════════════════════════════════
section("3 · Visualise — Normal vs Pneumonia")
# ════════════════════════════════════════════════════════
n_show  = 4
classes = [0, 1]
cnames  = ["Normal", "Pneumonia"]

fig, axes = plt.subplots(len(classes) * 3, n_show, figsize=(n_show*4, len(classes)*3*4))
fig.patch.set_facecolor("#0F172A")
fig.suptitle("Grad-CAM Explainability — Normal vs Pneumonia",
             color="white", fontsize=15, fontweight="bold", y=1.01)

row_labels = ["Original", "Heatmap", "Overlay"]
colormap   = cmap_module.get_cmap("jet")

for ci, cls in enumerate(classes):
    idx = np.where(y_te == cls)[0][:n_show]
    for col, i in enumerate(idx):
        img_norm = X_te_p[i]                # normalised
        img_vis  = denormalise(img_norm).clip(0, 1)

        hmap     = synthetic_gradcam(img_vis, seed=i+cls*10)
        heat_rgb = colormap(hmap)[:, :, :3].astype(np.float32)
        overlay  = (img_vis * 0.55 + heat_rgb * 0.45).clip(0, 1)

        for row_off, (show_img, rl) in enumerate(
            zip([img_vis, heat_rgb, overlay], row_labels)
        ):
            ax = axes[ci*3 + row_off][col]
            ax.imshow(show_img)
            if col == 0:
                ax.set_ylabel(
                    f"{cnames[cls]}\n{rl}",
                    color="white", fontsize=9, rotation=90
                )
            ax.axis("off")

plt.tight_layout()
out = ROOT / "outputs" / "gradcam" / "gradcam_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"  Saved → {out}")


# ════════════════════════════════════════════════════════
section("4 · Clinical Relevance Notes")
# ════════════════════════════════════════════════════════
print("""
  WHY GRAD-CAM MATTERS IN MEDICAL AI
  ────────────────────────────────────────────────────────
  ● Regulatory Compliance:
    FDA & CE Mark guidelines require AI tools to be
    interpretable before clinical deployment.

  ● Physician Trust:
    Radiologists will not trust a "black box" model.
    Grad-CAM shows WHAT the model is looking at.

  ● Error Analysis:
    If the heatmap highlights the image border, the
    model is cheating — using artefacts, not anatomy.

  ● Dataset Bias Detection:
    Spurious correlations (e.g. scanner watermarks)
    become visible in the heatmap.

  ● Research Value:
    Grad-CAM outputs can be compared against
    radiologist annotations for validation studies.
  ────────────────────────────────────────────────────────
  ✅  Notebook 03 complete! Check outputs/gradcam/
""")
