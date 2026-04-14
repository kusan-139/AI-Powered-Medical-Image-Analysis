"""
notebooks/01_data_exploration.py
──────────────────────────────────────────────────────────
Run this script (or convert to .ipynb) to:
  1. Explore the Chest X-Ray dataset structure
  2. Visualise sample images per class
  3. Plot class distribution
  4. Apply and compare preprocessing steps
  5. Show CLAHE enhancement side-by-side
──────────────────────────────────────────────────────────
Execute: python notebooks/01_data_exploration.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.data_loader import generate_synthetic_dataset


def section(title):
    print("\n" + "=" * 56)
    print(f"  {title}")
    print("=" * 56)


# ════════════════════════════════════════════════════════
section("1 · Generate Synthetic Chest X-Ray Dataset")
# ════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = generate_synthetic_dataset(
    n_samples=300, img_size=(224, 224), n_classes=2
)
print(f"  Train: {X_train.shape}  Labels: {y_train.shape}")
print(f"  Test : {X_test.shape}   Labels: {y_test.shape}")
print(f"  Pixel range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"  Classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")


# ════════════════════════════════════════════════════════
section("2 · Class Distribution Bar Chart")
# ════════════════════════════════════════════════════════
classes, counts = np.unique(y_train, return_counts=True)
class_names = ["Normal", "Pneumonia"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0F172A")

ax = axes[0]
bars = ax.bar(class_names, counts, color=["#34D399", "#F97316"], edgecolor="none", width=0.6)
ax.set_facecolor("#1E293B")
ax.set_title("Class Distribution (Training Set)", color="white", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Samples", color="#94A3B8")
ax.tick_params(colors="#94A3B8")
for spine in ax.spines.values():
    spine.set_edgecolor("#334155")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(cnt), ha="center", va="bottom", color="white", fontweight="bold")

# Class-wise sample grid
ax2 = axes[1]
ax2.set_facecolor("#1E293B")
ax2.set_title("Pixel Intensity Distribution", color="white", fontsize=13, fontweight="bold")
for i, cls in enumerate(classes):
    mask = y_train == cls
    mean_intensities = X_train[mask].reshape(mask.sum(), -1).mean(axis=1)
    ax2.hist(mean_intensities, bins=30, alpha=0.7,
             color=["#34D399", "#F97316"][i], label=class_names[i])
ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")
ax2.set_xlabel("Mean Pixel Intensity", color="#94A3B8")
ax2.set_ylabel("Count", color="#94A3B8")
ax2.tick_params(colors="#94A3B8")

plt.tight_layout()
out = ROOT / "outputs" / "plots" / "data_exploration.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"  Saved → {out}")


# ════════════════════════════════════════════════════════
section("3 · Sample Images Per Class")
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.patch.set_facecolor("#0F172A")
fig.suptitle("Sample Synthetic Chest X-Ray Images", color="white",
             fontsize=14, fontweight="bold", y=1.01)

for row, cls in enumerate(classes):
    idx = np.where(y_train == cls)[0][:4]
    for col, i in enumerate(idx):
        ax = axes[row][col]
        ax.imshow(X_train[i], cmap="gray" if X_train[i].mean() < 0.3 else None)
        ax.set_title(
            f"{class_names[cls]} #{i}",
            color="white", fontsize=10
        )
        ax.axis("off")

plt.tight_layout()
out2 = ROOT / "outputs" / "plots" / "sample_images.png"
plt.savefig(out2, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"  Saved → {out2}")


# ════════════════════════════════════════════════════════
section("4 · CLAHE Preprocessing Comparison")
# ════════════════════════════════════════════════════════
try:
    from preprocessing.preprocess import apply_clahe_batch
    clahe_available = True
except Exception:
    clahe_available = False

if clahe_available:
    sample = X_train[:3]
    enhanced = apply_clahe_batch(sample)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.patch.set_facecolor("#0F172A")
    fig.suptitle("CLAHE Enhancement Comparison", color="white", fontsize=13, fontweight="bold")

    for col in range(3):
        for row, (img, title) in enumerate([
            (sample[col],   "Original"),
            (enhanced[col], "CLAHE Enhanced"),
        ]):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{title} #{col}", color="white", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    out3 = ROOT / "outputs" / "plots" / "clahe_comparison.png"
    plt.savefig(out3, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out3}")
else:
    print("  CLAHE skipped (OpenCV not available).")


# ════════════════════════════════════════════════════════
section("5 · Summary Statistics")
# ════════════════════════════════════════════════════════
print(f"  Dataset:               Synthetic Chest X-Ray")
print(f"  Total training images: {len(X_train)}")
print(f"  Total test images    : {len(X_test)}")
print(f"  Image shape          : {X_train[0].shape}")
print(f"  Class balance        : {dict(zip(class_names, counts))}")
print(f"  Mean pixel value     : {X_train.mean():.4f}")
print(f"  Std  pixel value     : {X_train.std():.4f}")
print()
print("  ✅  Notebook 01 complete! Check outputs/plots/")
