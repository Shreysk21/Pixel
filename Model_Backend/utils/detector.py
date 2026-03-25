from __future__ import annotations

from typing import Dict, List

from PIL import Image, ImageDraw
from ultralytics import YOLO


Detection = Dict[str, object]


class XRayDetector:
    """YOLOv8 wrapper for X-ray detection and visualization."""

    def __init__(self, model_path: str = "model/best.pt") -> None:
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Run YOLOv8 inference on a PIL image and return normalized detections.

        Each detection includes:
        - label: class name
        - confidence: float in [0, 1]
        - bbox: [x1, y1, x2, y2] pixel coordinates
        """
        results = self.model(image, conf=0.25)
        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections

        class_names = result.names if result.names is not None else {}

        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()

        for bbox, confidence, class_id in zip(xyxy, confs, class_ids):
            class_idx = int(class_id)
            label = str(class_names.get(class_idx, str(class_idx)))
            detections.append(
                {
                    "label": label,
                    "confidence": float(confidence),
                    "bbox": [float(coord) for coord in bbox],
                }
            )

        return detections

    def draw_boxes(self, image: Image.Image, detections: List[Detection]) -> Image.Image:
        """
        Draw colored boxes and labels on a copy of the input PIL image.
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for detection in detections:
            label = str(detection.get("label", "unknown"))
            confidence = float(detection.get("confidence", 0.0))
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [float(v) for v in bbox]  # type: ignore[arg-type]

            color = self._color_for_label(label)

            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

            text = f"{label} {confidence * 100:.1f}%"
            text_x = x1
            text_y = max(0.0, y1 - 14.0)

            draw.text((text_x, text_y), text, fill=color)

        return annotated

    @staticmethod
    def _color_for_label(label: str) -> str:
        normalized = label.strip().lower()
        prohibited = {"gun", "bullet", "knife"}
        suspicious = {
            "baton",
            "plier",
            "hammer",
            "powerbank",
            "scissors",
            "wrench",
            "sprayer",
            "handcuffs",
            "lighter",
        }

        if normalized in prohibited:
            return "red"
        if normalized in suspicious:
            return "orange"
        return "green"
