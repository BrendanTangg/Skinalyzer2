"""
Make a training set that looks like what the app's model sees: every ACNE04-v2 photo
cropped to the padded face box exactly as app_crop.py / the app service does it.

Input:  datasets/acne04v2_yolo/   (from prepare_acne04v2.py)
Output: datasets/acne04v2_crops/images|labels/{train,val,test}/  and  acne04v2_crops.yaml

Boxes are moved into crop coordinates; boxes whose centre falls outside the crop are
dropped (the app drops those too). No face found -> the whole photo is kept, like the app.

Usage:
    python prepare_app_crops.py
"""

import shutil
from multiprocessing import Pool
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

from app_crop import LANDMARKER, PAD_BOTTOM, PAD_SIDE, PAD_TOP, load_rgb

HERE = Path(__file__).resolve().parent
SRC = HERE / "datasets/acne04v2_yolo"
OUT = HERE / "datasets/acne04v2_crops"
YAML = HERE / "acne04v2_crops.yaml"

_landmarker = None


def face_box(rgb):
    global _landmarker
    if _landmarker is None:
        _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1))
    h, w = rgb.shape[:2]
    found = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not found.face_landmarks:
        return None
    xs = [lm.x * w for lm in found.face_landmarks[0]]
    ys = [lm.y * h for lm in found.face_landmarks[0]]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    return (int(max(0, x1 - PAD_SIDE * bw)), int(max(0, y1 - PAD_TOP * bh)),
            int(min(w, x2 + PAD_SIDE * bw)), int(min(h, y2 + PAD_BOTTOM * bh)))


def do_one(job):
    part, img_path = job
    rgb = load_rgb(img_path)
    h, w = rgb.shape[:2]
    box = face_box(rgb)
    found = box is not None
    x1, y1, x2, y2 = box if found else (0, 0, w, h)
    cw, ch = x2 - x1, y2 - y1
    Image.fromarray(np.ascontiguousarray(rgb[y1:y2, x1:x2])).save(
        OUT / "images" / part / img_path.name, quality=95)

    lines, dropped = [], 0
    label_file = SRC / "labels" / part / (img_path.stem + ".txt")
    for line in label_file.read_text().split("\n"):
        if not line.strip():
            continue
        _, xc, yc, bw, bh = map(float, line.split())
        cx, cy, bw, bh = xc * w - x1, yc * h - y1, bw * w, bh * h
        if not (0 <= cx < cw and 0 <= cy < ch):
            dropped += 1
            continue
        bx1, by1 = max(0, cx - bw / 2), max(0, cy - bh / 2)
        bx2, by2 = min(cw, cx + bw / 2), min(ch, cy + bh / 2)
        lines.append(f"0 {(bx1 + bx2) / 2 / cw:.6f} {(by1 + by2) / 2 / ch:.6f} {(bx2 - bx1) / cw:.6f} {(by2 - by1) / ch:.6f}")
    (OUT / "labels" / part / (img_path.stem + ".txt")).write_text("\n".join(lines) + "\n")
    return part, found, len(lines), dropped, cw, ch


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    jobs = []
    for part in ("train", "val", "test"):
        (OUT / "images" / part).mkdir(parents=True)
        (OUT / "labels" / part).mkdir(parents=True)
        jobs += [(part, p) for p in sorted((SRC / "images" / part).iterdir())]

    with Pool(4) as pool:
        results = pool.map(do_one, jobs, chunksize=8)

    YAML.write_text(
        "# Acne04-v2 photos cropped to the face like the app does (prepare_app_crops.py)\n"
        "# Research use only: ACNE04 is 'free for academic usage'; Acne04-v2 states no license.\n"
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: acne\n")

    lines = ["Acne04-v2 -> app-style face crops"]
    for part in ("train", "val", "test"):
        rs = [r for r in results if r[0] == part]
        lines.append(f"{part:5s}: {len(rs)} photos, {sum(r[2] for r in rs)} boxes kept, "
                     f"{sum(r[3] for r in rs)} dropped (outside face crop), no face found in {sum(not r[1] for r in rs)}")
    sizes = sorted(max(r[4], r[5]) for r in results)
    lines.append(f"crop long side: median {sizes[len(sizes) // 2]} px (min {sizes[0]}, max {sizes[-1]})")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
