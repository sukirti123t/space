"""
train_yolo.py
--------------
Fine-tunes a YOLOv8n model on a space-debris imagery dataset (e.g. the
Kaggle "space-debris-detection-dataset-for-yolov8" dataset) and saves
the result to models/yolov8_debris.pt for use by detect.py.

--------------------------------------------------------------------
STEP 1 — Get Kaggle API credentials
--------------------------------------------------------------------
1. Go to https://www.kaggle.com/settings -> "Create New Token"
2. This downloads kaggle.json. Place it at ~/.kaggle/kaggle.json
   (chmod 600 ~/.kaggle/kaggle.json)

--------------------------------------------------------------------
STEP 2 — Download the dataset
--------------------------------------------------------------------
    pip install kaggle
    kaggle datasets download -d <owner>/space-debris-detection-dataset-for-yolov8 -p data/raw --unzip

NOTE: Verify the exact dataset slug on Kaggle before running (dataset
slugs/owners occasionally change) — search "space debris yolov8" on
kaggle.com/datasets and update DATASET_SLUG below accordingly.

--------------------------------------------------------------------
STEP 3 — Confirm dataset structure & data.yaml
--------------------------------------------------------------------
Most YOLOv8-ready Kaggle debris datasets already ship a data.yaml with
train/val image+label folders in YOLO format:

    data/raw/
    ├── data.yaml
    ├── train/images, train/labels
    └── val/images,   val/labels

If the dataset only provides a flat images+annotations folder, use
Ultralytics' auto-split utilities or a tool like Roboflow to convert
into this structure before training.

--------------------------------------------------------------------
STEP 4 — Train
--------------------------------------------------------------------
    python train_yolo.py --data data/raw/data.yaml --epochs 50 --imgsz 640

The best checkpoint is automatically copied to models/yolov8_debris.pt
when training completes.

Author: BSERC Internship Project
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for space debris detection.")
    parser.add_argument("--data", type=str, default="data/raw/data.yaml",
                         help="Path to the dataset's data.yaml (YOLO format).")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt",
                         help="Base pretrained checkpoint to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None,
                         help="'cpu', '0' (GPU 0), etc. Auto-detected if omitted.")
    parser.add_argument("--project", type=str, default="runs/debris_train")
    parser.add_argument("--name", type=str, default="yolov8_debris")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: data.yaml not found at '{args.data}'.")
        print("Download the dataset first -- see the module docstring in train_yolo.py "
              "for Kaggle download instructions.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"Loading base model: {args.base_model}")
    model = YOLO(args.base_model)

    print(
        f"Starting training: epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, device={args.device or 'auto'}"
    )
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        dest = os.path.join("models", "yolov8_debris.pt")
        shutil.copy(best_weights, dest)
        print(f"\n✅ Training complete. Best weights copied to: {dest}")
        print("Restart the Streamlit app to use your custom-trained debris model.")
    else:
        print(f"\n⚠️ Training finished but best.pt not found at expected path: {best_weights}")
        print("Check the runs/ directory for your training output.")


if __name__ == "__main__":
    main()
