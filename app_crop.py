"""
Run the acne model exactly the way the BlueBerry app's service does (task brief, section 1):

  a. Decode the JPEG, apply EXIF rotation, convert to RGB.
  b. MediaPipe Face Landmarker finds the face. Box around all landmarks, padded
     6% each side, 10% top, 4% bottom.
  c. Crop the image to that box at full resolution. No face found -> whole image.
  d. YOLO(weights).predict(source=crop, conf=0.25, imgsz=1024)
  e. Map boxes back to the full image; drop boxes whose centre is outside the face box.

Assumption (the brief doesn't say): the padding percentages are of the face box's
own width (sides) and height (top/bottom). Change PAD_* below if the app does it differently.

Open question for the app team: Ultralytics treats a numpy array as BGR (OpenCV order).
If the service passes its RGB crop to predict() as a numpy array, the model sees red and
blue swapped. AppPipeline(swap_red_blue=True) reproduces that, so both can be measured.
The default (False) gives the model the colours it was trained on.

Use from other scripts:
    from app_crop import AppPipeline
    pipe = AppPipeline("skinalyzer2.pt")
    result = pipe.run("photo.jpg")   # result.boxes: list of (x1, y1, x2, y2, conf) in full-image pixels

Or on its own, to see what the model gets:
    python app_crop.py captures/EXP-001            # saves crops + boxes to runs/app_crop/
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
LANDMARKER = HERE / "models/face_landmarker.task"
CONF = 0.25
IMGSZ = 1024
PAD_SIDE, PAD_TOP, PAD_BOTTOM = 0.06, 0.10, 0.04
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_rgb(path):
    """Step a: decode, apply EXIF rotation, convert to RGB. Returns an HxWx3 uint8 array."""
    with Image.open(path) as im:
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))


@dataclass
class AppResult:
    path: str
    width: int
    height: int
    face_found: bool
    face_box: tuple            # (x1, y1, x2, y2) the crop, in full-image pixels
    boxes: list = field(default_factory=list)   # (x1, y1, x2, y2, conf), full-image pixels, after step e
    ms: float = 0.0            # time for steps b-e


class AppPipeline:
    def __init__(self, weights=HERE / "skinalyzer2.pt", conf=CONF, imgsz=IMGSZ, device="cpu",
                 swap_red_blue=False):
        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.swap_red_blue = swap_red_blue
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1)
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def face_box(self, rgb):
        """Step b: padded box around all face landmarks, or None if no face."""
        h, w = rgb.shape[:2]
        found = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not found.face_landmarks:
            return None
        xs = [lm.x * w for lm in found.face_landmarks[0]]
        ys = [lm.y * h for lm in found.face_landmarks[0]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        bw, bh = x2 - x1, y2 - y1
        return (int(max(0, x1 - PAD_SIDE * bw)), int(max(0, y1 - PAD_TOP * bh)),
                int(min(w, x2 + PAD_SIDE * bw)), int(min(h, y2 + PAD_BOTTOM * bh)))

    def run(self, path):
        rgb = load_rgb(path)
        h, w = rgb.shape[:2]
        start = time.perf_counter()
        box = self.face_box(rgb)
        face_found = box is not None
        if not face_found:
            box = (0, 0, w, h)                                       # step c fallback
        x1, y1, x2, y2 = box
        crop = rgb[y1:y2, x1:x2]                                     # step c
        # Ultralytics treats numpy arrays as BGR, so flip RGB -> BGR to hand it the right colours
        # (unless reproducing a service that passes the RGB array as-is)
        feed = crop if self.swap_red_blue else crop[:, :, ::-1]
        pred = self.model.predict(source=np.ascontiguousarray(feed), conf=self.conf,
                                  imgsz=self.imgsz, device=self.device, verbose=False)[0]   # step d
        boxes = []
        for (bx1, by1, bx2, by2), c in zip(pred.boxes.xyxy.tolist(), pred.boxes.conf.tolist()):
            fx1, fy1, fx2, fy2 = bx1 + x1, by1 + y1, bx2 + x1, by2 + y1   # step e: back to full image
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                boxes.append((fx1, fy1, fx2, fy2, c))
        ms = (time.perf_counter() - start) * 1000
        return AppResult(str(path), w, h, face_found, box, boxes, ms)


def main():
    import cv2
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "captures/EXP-001"
    out = HERE / "runs/app_crop" / src.name
    out.mkdir(parents=True, exist_ok=True)
    pipe = AppPipeline()
    for path in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        r = pipe.run(path)
        img = cv2.cvtColor(load_rgb(path), cv2.COLOR_RGB2BGR)
        t = max(2, r.width // 500)
        cv2.rectangle(img, r.face_box[:2], r.face_box[2:], (255, 0, 0), t * 2)
        for bx1, by1, bx2, by2, c in r.boxes:
            cv2.rectangle(img, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 200, 0), t)
        cv2.imwrite(str(out / path.name), img)
        print(f"{path.name}: {len(r.boxes)} acne, face {'found' if r.face_found else 'NOT found'}, "
              f"crop {r.face_box[2] - r.face_box[0]}x{r.face_box[3] - r.face_box[1]}, {r.ms:.0f} ms")
    print(f"\nSaved to {out}  (blue = face crop, green = acne boxes)")


if __name__ == "__main__":
    main()
