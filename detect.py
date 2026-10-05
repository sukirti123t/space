"""
detect.py
---------
YOLOv8-based space debris / object detection module.

Designed to run inference on telescope, radar-derived, or simulated
space-imagery frames and return bounding boxes for candidate debris
objects. If a custom-trained weights file (models/yolov8_debris.pt)
is not present, the module automatically falls back to a stock
Ultralytics COCO-pretrained model so the app remains fully demoable
out of the box -- with a clear on-screen notice that results are
illustrative until a custom debris model is trained.

To train your own debris-detection model:
    1. Collect/label a dataset (e.g. synthetic star-field + debris
       renders, or the "SPARK" / "SPEED+" datasets) in YOLO format.
    2. yolo detect train data=debris.yaml model=yolov8n.pt epochs=100 imgsz=640
    3. Copy the resulting best.pt to models/yolov8_debris.pt

Author: BSERC Internship Project
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

MODEL_PATH_CUSTOM = os.path.join("models", "yolov8_debris.pt")
MODEL_PATH_FALLBACK = "yolov8n.pt"  # auto-downloaded by ultralytics on first use

BOX_COLOR = (0, 255, 200)      # neon cyan, BGR
TEXT_COLOR = (10, 10, 10)
LABEL_BG = (0, 255, 200)


@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]


class DebrisDetector:
    """Wraps an Ultralytics YOLOv8 model with graceful degradation."""

    def __init__(self, model_path: str = MODEL_PATH_CUSTOM):
        self.model = None
        self.using_fallback = False
        self.load_error = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self.load_error = (
                "The 'ultralytics' package is not installed. "
                "Run: pip install ultralytics"
            )
            return

        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.using_fallback = False
            else:
                # Fall back to a general pretrained model so the demo still runs.
                self.model = YOLO(MODEL_PATH_FALLBACK)
                self.using_fallback = True
        except Exception as e:
            self.load_error = f"Failed to load YOLO model: {e}"
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def detect(self, image, conf_threshold: float = 0.25) -> Tuple[np.ndarray, List[Detection]]:
        """
        Run detection on a PIL.Image, numpy array (BGR/RGB), or file path.
        Returns (annotated_image_rgb_ndarray, list_of_Detection)
        """
        if not self.is_ready:
            raise RuntimeError(self.load_error or "Detector not initialized.")

        img_bgr = self._to_bgr_ndarray(image)

        results = self.model.predict(
            source=img_bgr, conf=conf_threshold, verbose=False
        )
        result = results[0]

        detections: List[Detection] = []
        annotated = img_bgr.copy()

        names = result.names if hasattr(result, "names") else {}

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

            if self.using_fallback:
                # Relabel generic COCO classes as "possible debris candidate"
                # since no domain-specific debris class exists in the stock model.
                display_label = "debris-candidate"
            else:
                display_label = label

            x1, y1, x2, y2 = xyxy
            detections.append(Detection(display_label, conf, (x1, y1, x2, y2)))
            self._draw_box(annotated, (x1, y1, x2, y2), display_label, conf)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return annotated_rgb, detections

    @staticmethod
    def _draw_box(img_bgr, box, label, conf):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_COLOR, 2)
        caption = f"{label} {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), LABEL_BG, -1)
        cv2.putText(img_bgr, caption, (x1 + 3, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    @staticmethod
    def _to_bgr_ndarray(image) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image from path: {image}")
            return img
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:
                # Assume RGB coming from Streamlit/PIL pipelines
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")


def simulate_synthetic_frame(width: int = 640, height: int = 480, n_objects: int = 5,
                              seed: int = None) -> Image.Image:
    """
    Generates a synthetic star-field frame with bright dot 'objects' for
    offline demos when no real dataset/image is available. Useful as a
    quick sanity check of the detection pipeline in front of judges.
    """
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Starfield noise
    n_stars = 300
    xs = rng.integers(0, width, n_stars)
    ys = rng.integers(0, height, n_stars)
    for x, y in zip(xs, ys):
        brightness = int(rng.integers(80, 255))
        cv2.circle(frame, (int(x), int(y)), 1, (brightness, brightness, brightness), -1)

    # Bright "debris-like" objects
    for _ in range(n_objects):
        x, y = rng.integers(40, width - 40), rng.integers(40, height - 40)
        r = int(rng.integers(3, 8))
        cv2.circle(frame, (int(x), int(y)), r, (255, 255, 255), -1)
        cv2.circle(frame, (int(x), int(y)), r + 4, (150, 150, 200), 1)

    return Image.fromarray(frame)
