from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


OUTPUT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (768, 512)  # width, height


def _base_xray_canvas() -> Image.Image:
    width, height = IMAGE_SIZE
    base = np.full((height, width), 145, dtype=np.uint8)
    noise = np.random.normal(0, 7, size=(height, width)).astype(np.int16)
    canvas = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(canvas, mode="L")

    draw = ImageDraw.Draw(img)
    # Simulate benign cargo blocks.
    draw.rectangle((70, 90, 250, 220), fill=165)
    draw.rectangle((300, 120, 470, 250), fill=158)
    draw.rectangle((520, 80, 700, 240), fill=170)
    draw.rectangle((110, 300, 320, 430), fill=162)
    draw.rectangle((380, 290, 690, 450), fill=168)
    return img


def _create_clear_image() -> Image.Image:
    return _base_xray_canvas()


def _create_suspicious_image() -> Image.Image:
    img = _base_xray_canvas()
    draw = ImageDraw.Draw(img)
    # Knife-like elongated dark object near corner.
    draw.polygon([(560, 390), (735, 455), (725, 480), (545, 412)], fill=40)
    return img


def _create_prohibited_image() -> Image.Image:
    img = _base_xray_canvas()
    draw = ImageDraw.Draw(img)
    # Firearm-like L shape made of two connected bars.
    draw.rectangle((500, 230, 700, 280), fill=35)
    draw.rectangle((640, 280, 700, 410), fill=35)
    draw.rectangle((575, 280, 635, 340), fill=55)
    return img


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _create_clear_image().save(OUTPUT_DIR / "test_clear.png")
    _create_suspicious_image().save(OUTPUT_DIR / "test_suspicious.png")
    _create_prohibited_image().save(OUTPUT_DIR / "test_prohibited.png")

    print("Generated demo images:")
    print(f"- {OUTPUT_DIR / 'test_clear.png'}")
    print(f"- {OUTPUT_DIR / 'test_suspicious.png'}")
    print(f"- {OUTPUT_DIR / 'test_prohibited.png'}")


if __name__ == "__main__":
    main()
