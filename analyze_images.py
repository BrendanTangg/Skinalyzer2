"""
Run Skinalyzer on image files instead of the webcam.

Same acne model and analyzers as main3.py. For every image it saves an
annotated copy and adds its numbers to results.csv in the output folder.

Usage:
    python analyze_images.py photo.jpg
    python analyze_images.py photos/                # every image in a folder
    python analyze_images.py a.jpg b.png --lines --heatmap --out my_results
"""

import argparse
import csv
import os

from ultralytics import YOLO
import cv2
import mediapipe as mp

from skin_redness import RednessAnalyzer
from skin_pigmentation import PigmentationAnalyzer
from skin_oiliness import OilinessAnalyzer
from skin_texture import TextureAnalyzer

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
model_path = 'models/face_landmarker.task'


def score_color(score):
    """Green -> yellow -> red (BGR) as the score rises."""
    if score < 20:
        return (0, 200, 0)
    if score < 40:
        return (0, 200, 255)
    return (0, 0, 255)


def draw_label(img, text, x, y, color, scale=0.5):
    thick = max(1, int(round(scale * 2)))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def add_header(img, lines, scale):
    """Put the summary lines on a black band above the image, so they never
    overlap the acne boxes."""
    step = int(26 * scale)
    band = cv2.copyMakeBorder(img, step * len(lines) + int(10 * scale), 0, 0, 0,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for i, (text, color) in enumerate(lines):
        draw_label(band, text, 10, step * (i + 1), color, scale)
    return band


def face_regions(face_landmarks, w, h):
    """Cheek and forehead boxes, same placement as main3.py."""
    xs = [int(lm.x * w) for lm in face_landmarks]
    ys = [int(lm.y * h) for lm in face_landmarks]
    face_width = max(xs) - min(xs)
    face_height = max(ys) - min(ys)

    def box(lm, dx, dy, half_w, half_h):
        cx, cy = int(lm.x * w) + dx, int(lm.y * h) + dy
        return (max(0, cx - half_w), max(0, cy - half_h),
                min(w, cx + half_w), min(h, cy + half_h))

    cheek_dy = int(face_height * 0.07)
    cheek_w, cheek_h = int(face_width * 0.12), int(face_height * 0.15)
    return {
        "left_cheek": box(face_landmarks[50], 0, cheek_dy, cheek_w, cheek_h),
        "right_cheek": box(face_landmarks[280], 0, cheek_dy, cheek_w, cheek_h),
        "forehead": box(face_landmarks[10], 0, int(face_height * 0.04),
                        int(face_width * 0.32), int(face_height * 0.08)),
    }


def collect_images(inputs):
    paths = []
    for p in inputs:
        if os.path.isdir(p):
            paths += sorted(os.path.join(p, f) for f in os.listdir(p)
                            if f.lower().endswith(IMAGE_EXTS))
        else:
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Skinalyzer on image files")
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--out", default="outputs", help="folder for annotated images + results.csv")
    ap.add_argument("--conf", type=float, default=0.2, help="acne detection confidence")
    ap.add_argument("--heatmap", action="store_true", help="draw the redness heatmap")
    ap.add_argument("--lines", action="store_true", help="draw texture lines")
    ap.add_argument("--no-spots", action="store_true", help="don't outline pigment spots")
    ap.add_argument("--no-shine", action="store_true", help="don't tint shiny areas")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    acne_model = YOLO("skinalyzer2.pt")

    # smoothing=1.0: no averaging across images (that's only useful for video)
    redness = RednessAnalyzer(threshold=8.0, smoothing=1.0)
    pigmentation = PigmentationAnalyzer(smoothing=1.0)
    oiliness = OilinessAnalyzer(brightness=15.0, sat_drop=0.75, smoothing=1.0)
    texture = TextureAnalyzer(smoothing=1.0)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1)

    rows = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for path in collect_images(args.inputs):
            raw_frame = cv2.imread(path)
            if raw_frame is None:
                print(f"skip (can't read): {path}")
                continue
            h, w = raw_frame.shape[:2]
            name = os.path.splitext(os.path.basename(path))[0]
            # Text and lines scale with the image so big photos stay readable
            ts = max(0.5, w / 1200)

            acne_results = acne_model.predict(source=raw_frame, conf=args.conf, verbose=False)
            acne_boxes = acne_results[0].boxes.xyxy.cpu().numpy()
            frame = acne_results[0].plot()

            row = {"image": os.path.basename(path), "acne_count": len(acne_boxes)}

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
            face_result = landmarker.detect(mp_image)
            if not face_result.face_landmarks:
                print(f"{row['image']}: {row['acne_count']} acne, no face found (skin analysis skipped)")
                frame = add_header(frame, [(f"Acne: {row['acne_count']}  (no face found, skin analysis skipped)",
                                            (255, 255, 255))], ts)
                cv2.imwrite(os.path.join(args.out, name + "_analyzed.jpg"), frame)
                rows.append(row)
                continue

            face_landmarks = face_result.face_landmarks[0]
            regions = face_regions(face_landmarks, w, h)
            texture_exclude = None
            header = [(f"Acne: {row['acne_count']}", (255, 255, 255))]

            red = redness.analyze(raw_frame, face_landmarks, regions, acne_boxes)
            if red is not None:
                red_results, excess, skin = red
                if args.heatmap:
                    redness.draw_heatmap(frame, excess, skin)
                f = red_results["face"]
                row.update(redness=f["score"], red_pct=f["red_pct"], baseline_a=f["baseline_a"])
                header.append((f"Redness: {f['score']:.0f}  ({f['red_pct']:.1f}% area)",
                               score_color(f["score"])))

            pig = pigmentation.analyze(raw_frame, face_landmarks, regions, acne_boxes)
            if pig is not None:
                pig_results, spot_mask = pig
                texture_exclude = spot_mask
                if not args.no_spots:
                    pigmentation.draw_spots(frame, spot_mask)
                f = pig_results["face"]
                row.update(pigmentation=f["score"], spots=f["spots"],
                           spot_pct=f["spot_pct"], uneven=f["uneven"])
                header.append((f"Pigmentation: {f['score']:.0f}  spots: {f['spots']:.0f}  "
                               f"uneven: {f['uneven']:.2f}", score_color(f["score"])))

            oil = oiliness.analyze(raw_frame, face_landmarks, regions, acne_boxes)
            if oil is not None:
                oil_results, shine_mask = oil
                texture_exclude = (shine_mask if texture_exclude is None
                                   else texture_exclude | shine_mask)
                if not args.no_shine:
                    oiliness.draw_shine(frame, shine_mask)
                f = oil_results["face"]
                tz = oil_results.get("t_zone")
                row.update(oiliness=f["score"], shine_pct=f["shine_pct"],
                           t_zone_oiliness=tz["score"] if tz else "")
                header.append((f"Oiliness: {f['score']:.0f}  ({f['shine_pct']:.1f}% shiny)",
                               score_color(f["score"])))

            tex = texture.analyze(raw_frame, face_landmarks, regions, acne_boxes,
                                  exclude=texture_exclude)
            if tex is not None:
                tex_results, line_map, tex_skin = tex
                if args.lines:
                    texture.draw_lines(frame, line_map, tex_skin)
                f = tex_results["face"]
                row.update(texture=f["score"], lines=f["lines"], rough=f["rough"])
                header.append((f"Texture: {f['score']:.0f}  lines: {f['lines']:.1f}  "
                               f"rough: {f['rough']:.1f}", score_color(f["score"])))

            # Per-region scores next to each box
            per_region = [("red", red), ("pig", pig), ("oil", oil), ("tex", tex)]
            for rname, (x1, y1, x2, y2) in regions.items():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), max(2, int(2 * ts)))
                for i, (short, res) in enumerate(per_region):
                    if res is not None and rname in res[0]:
                        s = res[0][rname]["score"]
                        row[f"{rname}_{short}"] = s
                        draw_label(frame, f"{short} {s:.0f}", x1, y2 + int((15 + 15 * i) * ts),
                                   score_color(s), ts * 0.8)

            frame = add_header(frame, header, ts)
            out_path = os.path.join(args.out, name + "_analyzed.jpg")
            cv2.imwrite(out_path, frame)
            rows.append(row)
            print(f"{row['image']}: acne {row['acne_count']}, "
                  f"redness {row.get('redness', 0):.0f}, pigmentation {row.get('pigmentation', 0):.0f}, "
                  f"oiliness {row.get('oiliness', 0):.0f}, texture {row.get('texture', 0):.0f}")

    if rows:
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        with open(os.path.join(args.out, "results.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} annotated images + results.csv to {args.out}/")
    else:
        print("No images found.")


if __name__ == "__main__":
    main()
