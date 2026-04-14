# -*- coding: utf-8 -*-
"""
============================================================
AI-Powered Medical Image Analysis System
main.py - Unified CLI entry point
============================================================
Author  : Kusan Chakraborty
GitHub  : https://github.com/kusan-139
Project : AI Medical Image Analysis
============================================================"""

import io
import sys as _sys
# Force UTF-8 stdout on Windows so emoji/box-drawing prints correctly
if hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import sys
import os
import time
from pathlib import Path

# ── Make sure src/ is importable ──────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from utils.logger      import get_logger
from utils.config      import load_config

logger = get_logger("Main")

# ─────────────────────────────────────────────────────────
BANNER = """
+----------------------------------------------------------+
|   [AI]  AI-Powered Medical Image Analysis System  [RX]   |
|                                                          |
|  Pneumonia Detection  |  Skin Lesion Classification      |
|  Brain Tumor Segmentation  |  Grad-CAM Explainability    |
|                                                          |
|  Author  : Kusan Chakraborty(Student Portfolio Project)  |
|  Dataset : Chest-X-Ray14  |  HAM10000  |  BraTS          |
+----------------------------------------------------------+
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Medical Image Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo
  python main.py --mode train   --task pneumonia
  python main.py --mode predict --task pneumonia --image data/sample/chest_xray.jpg
  python main.py --mode evaluate --task skin
  python main.py --mode dashboard
        """
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "train", "predict", "evaluate", "dashboard", "setup"],
        default="demo",
        help="Execution mode (default: demo)"
    )
    parser.add_argument(
        "--task",
        choices=["pneumonia", "skin", "brain", "all"],
        default="pneumonia",
        help="Analysis task to run (default: pneumonia)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
def run_demo():
    """Run a full demonstration using synthetic data."""
    logger.info("🚀  Starting DEMO mode — no real dataset required")
    print("\n" + "─" * 58)
    print("  DEMO MODE: Generating synthetic medical images …")
    print("─" * 58)

    from data.data_loader       import generate_synthetic_dataset
    from models.cnn_classifier  import build_cnn_model
    from preprocessing.preprocess import preprocess_pipeline
    from evaluation.metrics     import evaluate_model
    from explainability.gradcam import run_gradcam_demo

    # Step 1 – Synthetic data
    print("\n[1/5] 📦  Generating synthetic chest X-ray dataset …")
    X_train, X_test, y_train, y_test = generate_synthetic_dataset(
        n_samples=200, img_size=(224, 224), n_classes=2
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

    # Step 2 – Preprocessing
    print("\n[2/5] ⚙️   Preprocessing pipeline …")
    X_train_p, X_test_p = preprocess_pipeline(X_train, X_test)
    print("      Normalisation + CLAHE + Augmentation applied ✓")

    # Step 3 – Model
    print("\n[3/5] 🧠  Building CNN model (MobileNetV2 backbone) …")
    model = build_cnn_model(input_shape=(224, 224, 3), num_classes=2)
    model.summary(print_fn=lambda x: None)   # suppress verbose keras output
    print(f"      Total parameters: {model.count_params():,}")

    # Step 4 – Evaluate (with random weights for demo)
    print("\n[4/5] 📊  Running evaluation …")
    results = evaluate_model(model, X_test_p, y_test, demo=True)
    print(f"      Accuracy : {results['accuracy']:.4f}")
    print(f"      AUC-ROC  : {results['auc']:.4f}")
    print(f"      F1 Score : {results['f1']:.4f}")

    # Step 5 – Grad-CAM
    print("\n[5/5] 🔍  Grad-CAM heatmap visualisation …")
    run_gradcam_demo(model, X_test_p[:1])
    print("      Heatmap saved → outputs/gradcam/demo_heatmap.png ✓")

    print("\n" + "=" * 58)
    print("  ✅  DEMO complete!  Check the outputs/ folder.")
    print("=" * 58 + "\n")


# ─────────────────────────────────────────────────────────
def run_train(args):
    """Train a model on the selected task."""
    logger.info(f"🏋️  Training mode | task={args.task} | epochs={args.epochs}")
    from training.trainer import Trainer
    trainer = Trainer(
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        config=load_config(args.config)
    )
    trainer.run()


# ─────────────────────────────────────────────────────────
def run_predict(args):
    """Run inference on a single image."""
    if args.image is None:
        logger.error("--image path is required for predict mode")
        sys.exit(1)

    logger.info(f"🔮  Predicting on: {args.image}")
    from inference.predictor import Predictor
    predictor = Predictor(task=args.task)
    result = predictor.predict(args.image)
    print("\n" + "─" * 40)
    print(f"  Image   : {args.image}")
    print(f"  Task    : {args.task}")
    print(f"  Class   : {result['class']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print("─" * 40 + "\n")


# ─────────────────────────────────────────────────────────
def run_evaluate(args):
    """Evaluate a trained model and generate a PDF report."""
    logger.info(f"📈  Evaluation mode | task={args.task}")
    from evaluation.evaluator import run_full_evaluation
    run_full_evaluation(task=args.task)


# ─────────────────────────────────────────────────────────
def run_dashboard():
    """Launch the Flask web dashboard."""
    logger.info("🌐  Launching web dashboard on http://localhost:5000")
    from dashboard.app import create_app
    app = create_app()
    print("\n  🌐  Dashboard running at: http://localhost:5000")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=False, host="0.0.0.0", port=5000)


# ─────────────────────────────────────────────────────────
def run_setup():
    """Create all necessary folders and download sample data."""
    logger.info("🔧  Setting up project …")
    folders = [
        "data/raw/chest_xray",
        "data/raw/skin_lesion",
        "data/raw/brain_mri",
        "data/processed",
        "data/sample",
        "models/saved",
        "models/checkpoints",
        "outputs/predictions",
        "outputs/gradcam",
        "outputs/reports",
        "outputs/plots",
        "logs",
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✓  Created: {folder}/")
    print("\n  ✅  Setup complete! Now run: python main.py --mode demo\n")


# ─────────────────────────────────────────────────────────
def main():
    print(BANNER)
    args = parse_args()

    dispatch = {
        "demo"      : run_demo,
        "train"     : run_train,
        "predict"   : run_predict,
        "evaluate"  : run_evaluate,
        "dashboard" : run_dashboard,
        "setup"     : run_setup,
    }

    fn = dispatch[args.mode]
    # Some modes need args, some don't
    if args.mode in ("demo", "dashboard", "setup"):
        fn()
    else:
        fn(args)


if __name__ == "__main__":
    main()
