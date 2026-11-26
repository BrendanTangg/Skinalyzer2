"""
Turn the Acne04-v2 labels into a YOLO training dataset.

Inputs (already downloaded):
    datasets/acne04/Classification/JPEGImages/       the ACNE04 photos
    datasets/acne04/acne04v2/Acne04-v2_annotations.json   the v2 labels (circles)
    datasets/acne04/Detection/VOC2007/ImageSets/Main/      the original train/test lists

Output:
    datasets/acne04v2_yolo/
        images/{train,val,test}/   links to the photos (no copies, saves ~1 GB)
        labels/{train,val,test}/   one .txt per photo, YOLO format
        report.txt                 what was converted and what was skipped
    acne04v2.yaml                  point YOLO's train(data=...) at this

Each v2 label is a circle (center + radius). YOLO wants boxes, so each circle
becomes the square that just contains it: class 0 ("acne"), then center x,
center y, width, height, all as fractions of the photo size.

Labels that can't be right are skipped and counted in report.txt:
  - the circle's center is outside the photo
  - the radius is bigger than MAX_RADIUS_FRAC of the photo's shorter side
    (a pimple can't cover a big part of the face)

Split: the test set is the original ACNE04 test list (SPLIT_FOLD), so results
can be compared with the published papers. A random VAL_FRAC of the rest
becomes validation (fixed SEED, so it's the same every run).

Usage:
    python prepare_acne04v2.py
"""

import json
import random
import shutil
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / "datasets/acne04"
PHOTOS = SRC / "Classification/JPEGImages"
LABELS_JSON = SRC / "acne04v2/Acne04-v2_annotations.json"
SPLITS = SRC / "Detection/VOC2007/ImageSets/Main"
OUT = HERE / "datasets/acne04v2_yolo"
YAML = HERE / "acne04v2.yaml"

SPLIT_FOLD = 0          # which of the original 5 train/test lists to use (0-4)
VAL_FRAC = 0.15         # share of the training photos held back for validation
SEED = 0
MAX_RADIUS_FRAC = 0.05  # skip circles with radius > 5% of the photo's shorter side (~155 px here)


def read_list(path):
    """Photo names from an ACNE04 split file. Lines look like 'levle0_1.jpg  0'."""
    names = set()
    for line in path.read_text().splitlines():
        if line.strip():
            name = line.split()[0]
            names.add(name if name.endswith(".jpg") else name + ".jpg")
    return names


def link_or_copy(src, dst):
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main():
    data = json.loads(LABELS_JSON.read_text())
    spots = {}
    for a in data["annotations"]:
        spots.setdefault(a["image_id"], []).append(a)

    test_names = read_list(SPLITS / f"NNEW_test_{SPLIT_FOLD}.txt")

    # Decide the split for every v2 photo
    rest = sorted(im["file_name"] for im in data["images"] if im["file_name"] not in test_names)
    random.Random(SEED).shuffle(rest)
    n_val = round(len(rest) * VAL_FRAC)
    split_of = {name: "val" for name in rest[:n_val]}
    split_of.update({name: "train" for name in rest[n_val:]})
    split_of.update({im["file_name"]: "test" for im in data["images"] if im["file_name"] in test_names})

    if OUT.exists():
        shutil.rmtree(OUT)
    for part in ("train", "val", "test"):
        (OUT / "images" / part).mkdir(parents=True)
        (OUT / "labels" / part).mkdir(parents=True)

    counts = {p: {"photos": 0, "boxes": 0} for p in ("train", "val", "test")}
    skipped_outside, skipped_big, size_mismatch, missing = [], [], [], []

    for im in data["images"]:
        name, w, h = im["file_name"], im["width"], im["height"]
        photo = PHOTOS / name
        if not photo.exists():
            missing.append(name)
            continue
        with Image.open(photo) as pic:
            real_w, real_h = pic.size
        if (real_w, real_h) != (w, h):
            size_mismatch.append(f"{name}: labels say {w}x{h}, photo is {real_w}x{real_h}")
            continue

        part = split_of[name]
        max_r = MAX_RADIUS_FRAC * min(w, h)
        lines = []
        for a in spots.get(im["id"], []):
            x, y = a["coordinates"]
            r = a["radius"]
            if not (0 <= x < w and 0 <= y < h):
                skipped_outside.append(f"{name}: center ({x}, {y})")
                continue
            if r > max_r:
                skipped_big.append(f"{name}: radius {r}")
                continue
            # square around the circle, clipped to the photo
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(w, x + r), min(h, y + r)
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")

        link_or_copy(photo, OUT / "images" / part / name)
        (OUT / "labels" / part / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
        counts[part]["photos"] += 1
        counts[part]["boxes"] += len(lines)

    YAML.write_text(
        f"# Acne04-v2 labels on ACNE04 photos, made by prepare_acne04v2.py\n"
        f"# Research use only: ACNE04 is 'free for academic usage'; Acne04-v2 states no license.\n"
        f"path: {OUT}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: acne\n"
    )

    report = [
        "Acne04-v2 -> YOLO conversion",
        f"Split: test = original ACNE04 NNEW_test_{SPLIT_FOLD}; val = {VAL_FRAC:.0%} of the rest (seed {SEED})",
        "",
    ]
    for part, c in counts.items():
        report.append(f"{part:5s}: {c['photos']:4d} photos, {c['boxes']:5d} boxes")
    total_boxes = sum(c["boxes"] for c in counts.values())
    report += [
        f"total: {sum(c['photos'] for c in counts.values()):4d} photos, {total_boxes:5d} boxes "
        f"(of {len(data['annotations'])} labels in the file)",
        "",
        f"Skipped, center outside the photo: {len(skipped_outside)}",
        f"Skipped, radius > {MAX_RADIUS_FRAC:.0%} of shorter side: {len(skipped_big)}",
        f"Photos skipped, size doesn't match the labels: {len(size_mismatch)}",
        f"Photos missing from JPEGImages: {len(missing)}",
        "",
        "Details:",
        *skipped_outside, *skipped_big, *size_mismatch, *missing,
    ]
    (OUT / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report[:12]))
    print(f"\nDataset: {OUT}\nYAML:    {YAML}\nFull list of skipped labels: {OUT / 'report.txt'}")


if __name__ == "__main__":
    main()
