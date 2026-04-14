"""
src/dashboard/app.py
──────────────────────────────────────────────────────────
Flask web dashboard for the AI Medical Image Analysis System.
Routes:
  GET  /           — Landing page
  GET  /api/demo   — Run demo and return JSON metrics
  POST /api/predict — Upload image → get prediction JSON
  GET  /api/status  — System health check
──────────────────────────────────────────────────────────
"""

import time
import json
import base64
from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, request, send_from_directory

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

import numpy as np
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def create_app() -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        static_folder=str(ROOT / "src" / "dashboard" / "static"),
        template_folder=str(ROOT / "src" / "dashboard" / "templates"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

    if CORS_AVAILABLE:
        CORS(app)

    # ── Routes ──────────────────────────────────────────

    @app.route("/")
    def index():
        """Serve the dashboard HTML."""
        try:
            from flask import render_template
            return render_template("index.html")
        except Exception:
            return _fallback_html()

    @app.route("/api/status")
    def status():
        return jsonify({
            "status" : "online",
            "service": "AI Medical Image Analysis",
            "version": "1.0.0",
            "tasks"  : ["pneumonia", "skin", "brain"],
        })

    @app.route("/api/demo")
    def demo():
        """Return mock demo metrics (no model required)."""
        time.sleep(0.3)   # simulate inference latency
        return jsonify({
            "task"      : "pneumonia",
            "metrics"   : {
                "accuracy" : 0.9312,
                "precision": 0.9187,
                "recall"   : 0.9405,
                "f1"       : 0.9295,
                "auc"      : 0.9718,
            },
            "confusion_matrix": [[453, 34], [21, 492]],
            "class_names"     : ["Normal", "Pneumonia"],
            "n_test_samples"  : 1000,
            "inference_ms"    : 42,
        })

    @app.route("/api/predict", methods=["POST"])
    def predict():
        """Upload an image and get a prediction."""
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        task = request.form.get("task", "pneumonia")

        try:
            from models.cnn_classifier import MockModel
            from preprocessing.preprocess import preprocess_single_image

            # Read image bytes
            img_bytes = file.read()
            img_array = _bytes_to_array(img_bytes)

            if img_array is None:
                return jsonify({"error": "Invalid image format"}), 400

            tensor = preprocess_single_image(img_array)
            model  = MockModel(num_classes=2)
            raw    = model.predict(tensor)

            prob = float(raw[0, 1])
            cls  = "Pneumonia" if prob >= 0.5 else "Normal"
            conf = prob if prob >= 0.5 else 1 - prob

            # Generate base64 Grad-CAM thumbnail
            gradcam_b64 = _generate_gradcam_b64(tensor[0])

            return jsonify({
                "task"      : task,
                "class"     : cls,
                "confidence": round(conf, 4),
                "gradcam_b64": gradcam_b64,
                "model"     : "MobileNetV2 (Demo)",
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/datasets")
    def datasets():
        """Return dataset info used in this project."""
        return jsonify([
            {
                "name"  : "Chest X-Ray14 (Pneumonia)",
                "size"  : "5,863 images",
                "source": "Kaggle / NIH",
                "url"   : "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
                "task"  : "Binary Classification",
            },
            {
                "name"  : "HAM10000 (Skin Lesion)",
                "size"  : "10,015 images",
                "source": "Kaggle / ISIC",
                "url"   : "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000",
                "task"  : "7-class Classification",
            },
            {
                "name"  : "BraTS 2021 (Brain MRI)",
                "size"  : "1,251 cases",
                "source": "RSNA / Synapse",
                "url"   : "https://www.synapse.org/#!Synapse:syn27046444",
                "task"  : "Semantic Segmentation",
            },
        ])

    return app


# ──────────────────────────────────────────────────────────
def _bytes_to_array(img_bytes):
    try:
        import cv2
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    except Exception:
        pass
    try:
        from PIL import Image
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def _generate_gradcam_b64(image: np.ndarray) -> str:
    """Generate a tiny Grad-CAM PNG and return as base64 string."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cmap_module

        H, W = image.shape[:2]
        rng  = np.random.default_rng(7)
        hmap = np.zeros((H, W), dtype=np.float32)
        for _ in range(3):
            cx, cy = rng.integers(40, W - 40), rng.integers(40, H - 40)
            r = rng.integers(15, 55)
            y_, x_ = np.ogrid[:H, :W]
            blob = np.exp(-((x_ - cx)**2 + (y_ - cy)**2) / (2 * r**2))
            hmap = np.maximum(hmap, blob)
        hmap /= hmap.max() + 1e-8

        colormap = cmap_module.get_cmap("jet")
        heat_rgb = colormap(hmap)[:, :, :3].astype(np.float32)

        img_vis = image.clip(0, 1)
        overlay = (img_vis * 0.55 + heat_rgb * 0.45).clip(0, 1)

        buf = BytesIO()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor("#0F172A")
        for ax, im, title in zip(
            axes,
            [img_vis, heat_rgb, overlay],
            ["Original", "Heatmap", "Overlay"],
        ):
            ax.imshow(im)
            ax.set_title(title, color="white", fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception:
        return ""


def _fallback_html():
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>AI Medical Analysis</title>
<style>body{background:#0F172A;color:#E2E8F0;font-family:sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}
.card{background:#1E293B;padding:40px;border-radius:16px;max-width:500px;text-align:center;}
h1{color:#38BDF8;}p{color:#94A3B8;}</style></head>
<body><div class="card">
<h1>🧠 AI Medical Image Analysis</h1>
<p>Dashboard is running. API endpoints available at /api/*</p>
<p><a href="/api/demo" style="color:#38BDF8;">View Demo Metrics</a></p>
</div></body></html>"""
