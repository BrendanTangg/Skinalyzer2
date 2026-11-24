"""
Count acne only inside one face zone (forehead or cheek).

Same acne model and settings as predict.py (conf 0.25, imgsz 1024), but every
box whose center falls outside the zone is thrown away before counting.
The zone comes from the file name: forehead_*.jpg -> forehead, cheek_*.jpg ->
the cheek facing the camera. Other photos (e.g. chin) are skipped.

Annotated images go to runs/detect/<RUN>/:
    blue rectangle = zone, green boxes = counted, red boxes = thrown away.

Usage:
    python predict_zones.py
"""

import csv
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
PHOTOS = HERE / "captures/EXP-001"
RUN = "EXP-005"              # output folder name; change it for each new experiment
OUT = HERE / "runs/detect" / RUN
CONF = 0.25
IMGSZ = 1024

# Zone sizes, as fractions of the face's width/height in the photo.
# HALF_W / HALF_H = half the zone's width / height (so 0.20 -> 40% of the face wide).
# EXP-002 used face_regions() from analyze_images.py: cheek 0.12 x 0.15, forehead 0.32 x 0.08.
CHEEK_HALF_W = 0.20          # was 0.12: wider
CHEEK_HALF_H = 0.15          # same as before
CHEEK_SHIFT_DOWN = 0.07      # zone center sits this far below the cheekbone point
CHEEK_EXTRA_EAR = 0.08       # EXP-004: extend the cheek zone this much further toward the ear (one side only)
CHEEK_EXTRA_UP = 0.05        # EXP-004: extend the top of the cheek zone this much further up, toward the eye
FOREHEAD_HALF_W = 0.40       # EXP-005: was 0.32, wider on both sides
FOREHEAD_HALF_H = 0.13       # was 0.08: taller
FOREHEAD_SHIFT_DOWN = 0.09   # was 0.04: moved down so the extra height goes toward the eyebrows, not into the hair
FOREHEAD_EXTRA_UP = 0.05     # EXP-005: extend the top of the forehead zone this much further up, toward the hairline


def face_regions(face_landmarks, w, h):
    """Cheek and forehead zone rectangles (x1, y1, x2, y2) in pixels.

    Same placement as face_regions() in analyze_images.py, but with the sizes above.
    Landmark 50 / 280 = a point on each cheekbone, 10 = top middle of the forehead.
    """
    xs = [lm.x * w for lm in face_landmarks]
    ys = [lm.y * h for lm in face_landmarks]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)

    def box(lm, shift_down, half_w, half_h):
        cx = lm.x * w
        cy = lm.y * h + face_h * shift_down
        return (int(max(0, cx - face_w * half_w)), int(max(0, cy - face_h * half_h)),
                int(min(w, cx + face_w * half_w)), int(min(h, cy + face_h * half_h)))

    nose_x = face_landmarks[1].x * w  # landmark 1 = nose tip

    def cheek_box(lm):
        """Cheek box, stretched toward the ear and upward toward the eye."""
        x1, y1, x2, y2 = box(lm, CHEEK_SHIFT_DOWN, CHEEK_HALF_W, CHEEK_HALF_H)
        extra = int(face_w * CHEEK_EXTRA_EAR)
        if lm.x * w < nose_x:      # ear is on the left of this cheek in the photo
            x1 = max(0, x1 - extra)
        else:                      # ear is on the right
            x2 = min(w, x2 + extra)
        y1 = max(0, y1 - int(face_h * CHEEK_EXTRA_UP))
        return (x1, y1, x2, y2)

    def forehead_box(lm):
        """Forehead box, stretched upward toward the hairline."""
        x1, y1, x2, y2 = box(lm, FOREHEAD_SHIFT_DOWN, FOREHEAD_HALF_W, FOREHEAD_HALF_H)
        return (x1, max(0, y1 - int(face_h * FOREHEAD_EXTRA_UP)), x2, y2)

    return {
        "left_cheek": cheek_box(face_landmarks[50]),
        "right_cheek": cheek_box(face_landmarks[280]),
        "forehead": forehead_box(face_landmarks[10]),
    }


def pick_zone(name, landmarks, regions):
    """Choose which zone box to use for this photo, from its file name."""
    if name.startswith("forehead"):
        return "forehead", regions["forehead"]
    if name.startswith("cheek"):
        # Use the cheek facing the camera. MediaPipe's z is depth: smaller
        # z = closer to the camera. Landmark 50 is the "left_cheek" box's
        # anchor and 280 the "right_cheek" one (see face_regions).
        side = "left_cheek" if landmarks[50].z < landmarks[280].z else "right_cheek"
        return "cheek", regions[side]
    return None, None


def inside(box, zone):
    """True if the center of an acne box is inside the zone rectangle."""
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(HERE / "skinalyzer2.pt")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HERE / "models/face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1)

    rows = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for path in sorted(PHOTOS.glob("*.jpg")):
            name = path.stem
            img = cv2.imread(str(path))
            h, w = img.shape[:2]

            face = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                              data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not face.face_landmarks:
                print(f"{path.name}: no face found, skipped")
                continue
            landmarks = face.face_landmarks[0]
            zone_name, zone = pick_zone(name, landmarks, face_regions(landmarks, w, h))
            if zone is None:
                print(f"{path.name}: no zone for this photo, skipped")
                continue

            boxes = model.predict(img, conf=CONF, imgsz=IMGSZ, verbose=False)[0].boxes.xyxy.cpu().numpy()
            kept = [b for b in boxes if inside(b, zone)]

            # Draw: zone in blue, kept boxes green, thrown-away boxes red
            t = max(2, w // 500)
            cv2.rectangle(img, zone[:2], zone[2:], (255, 0, 0), t * 2)
            for b in boxes:
                color = (0, 200, 0) if inside(b, zone) else (0, 0, 255)
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, t)
            label = f"{zone_name}: {len(kept)} in zone / {len(boxes)} total"
            cv2.putText(img, label, (20, 30 * t), cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, (0, 0, 0), t * 3)
            cv2.putText(img, label, (20, 30 * t), cv2.FONT_HERSHEY_SIMPLEX, t * 0.6, (255, 255, 255), t)
            cv2.imwrite(str(OUT / path.name), img)

            rows.append({"image": path.name, "zone": zone_name,
                         "all_boxes": len(boxes), "in_zone": len(kept)})
            print(f"{path.name}: {len(kept)} in zone ({len(boxes)} total)")

    with open(OUT / "counts.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "zone", "all_boxes", "in_zone"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
