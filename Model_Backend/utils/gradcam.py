from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def overlay_heatmap(original_image: Image.Image, heatmap_array: np.ndarray) -> Image.Image:
    original_rgb = original_image.convert("RGB")
    original_np = np.array(original_rgb, dtype=np.float32) / 255.0
    heatmap_normalized = np.clip(heatmap_array.astype(np.float32), 0.0, 255.0)
    heatmap_uint8 = heatmap_normalized.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_np = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if heatmap_np.shape[:2] != original_np.shape[:2]:
        heatmap_np = cv2.resize(heatmap_np, (original_np.shape[1], original_np.shape[0]))
    blended = (0.6 * original_np) + (0.4 * heatmap_np)
    blended = np.uint8(np.clip(blended, 0.0, 1.0) * 255)
    return Image.fromarray(blended)


def generate_gradcam(
    model_path: str,
    image: Image.Image,
    target_class_idx: int = 0,
) -> Image.Image:
    try:
        yolo = YOLO(model_path)
        image_rgb = image.convert("RGB")
        original_w, original_h = image_rgb.size

        results = yolo(image_rgb, conf=0.25, verbose=False)

        heatmap = np.zeros((original_h, original_w), dtype=np.float32)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_w, x2)
                y2 = min(original_h, y2)
                conf = float(box.conf[0])
                heatmap[y1:y2, x1:x2] += conf

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_blurred = cv2.GaussianBlur(heatmap, (51, 51), 0)

        if heatmap_blurred.max() > 0:
            heatmap_blurred = heatmap_blurred / heatmap_blurred.max()

        heatmap_uint8 = np.uint8(255 * heatmap_blurred)
        return overlay_heatmap(image_rgb, heatmap_uint8)

    except Exception as exc:
        print(f"[GradCAM Warning] Failed: {exc}")
        return image
