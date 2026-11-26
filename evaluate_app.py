"""
Measure an acne model the way the app team asked (task brief, section 6), using the
app's own crop-and-predict steps (app_crop.py).

Reports:
  1. Precision and recall, separately for frontal and oblique views (and per angle).
  2. Repeatability: for each real pimple, in how many of the views where it is visible
     the model found it, and whether the app would keep it or mark it "inconsistent"
     (seen in 4+ views but found in fewer than one third of them).

A detection counts as correct when its CENTRE lies inside a labelled pimple's box
(plus MATCH_PAD pixels), because the box centre is what the app projects onto the 3D face.
Each label can be matched by one detection only; extra detections on it are false positives.

LABELS: a CSV with one row per pimple per photo it is visible in:

    image,view,lesion_id,x1,y1,x2,y2
    face1_front.jpg,frontal,A1,812,1033,834,1052
    face1_left30.jpg,left30,A1,640,1029,659,1047
    face1_left30.jpg,left30,A2,701,1210,720,1228
    face1_nolesions.jpg,frontal,,,,,          <- a photo with no pimples: blank lesion_id and box

  image      file name inside the images folder
  view       frontal, left30, right30, left60, right60 (anything else is reported as its own group)
  lesion_id  the SAME id for the same pimple in every photo (needed for repeatability)
  x1..y2     box around the pimple in full-image pixels, after EXIF rotation (upright photo)

Usage:
    python evaluate_app.py --images eval/sweep1 --labels eval/sweep1/labels.csv
    python evaluate_app.py --acne04v2-test            # sanity check on the ACNE04-v2 test split
    options: --weights other.pt  --conf 0.25  --imgsz 1024  --swap-red-blue  --draw
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from app_crop import HERE, AppPipeline, load_rgb

MATCH_PAD = 3          # px of slack around a label box when checking a detection centre
OBLIQUE = {"left30", "right30", "left60", "right60"}


def read_labels(path):
    """-> {image: {"view": str, "lesions": [(lesion_id, x1, y1, x2, y2), ...]}}"""
    images = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            entry = images.setdefault(row["image"], {"view": row["view"].strip() or "unknown", "lesions": []})
            if row["x1"].strip():
                entry["lesions"].append((row["lesion_id"].strip(),
                                         *(float(row[k]) for k in ("x1", "y1", "x2", "y2"))))
    return images


def acne04v2_test_labels():
    """The ACNE04-v2 test photos as a label set (one view each, so no repeatability)."""
    data = json.loads((HERE / "datasets/acne04/acne04v2/Acne04-v2_annotations.json").read_text())
    test_dir = HERE / "datasets/acne04v2_yolo/images/test"
    keep = {p.name for p in test_dir.iterdir()}
    spots = defaultdict(list)
    for a in data["annotations"]:
        spots[a["image_id"]].append(a)
    images = {}
    for im in data["images"]:
        if im["file_name"] not in keep:
            continue
        lesions = []
        for a in spots[im["id"]]:
            (x, y), r = a["coordinates"], a["radius"]
            if 0 <= x < im["width"] and 0 <= y < im["height"] and r <= 0.05 * min(im["width"], im["height"]):
                lesions.append((f'{im["file_name"]}#{a["id"]}', x - r, y - r, x + r, y + r))
        images[im["file_name"]] = {"view": "acne04", "lesions": lesions}
    return test_dir, images


def match(dets, lesions):
    """Greedy one-to-one match, most confident detection first.
    Returns (tp_dets, fp_dets, matched_lesion_ids)."""
    free = list(lesions)
    tp, fp, found = [], [], set()
    for d in sorted(dets, key=lambda d: -d[4]):
        cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
        hits = [l for l in free if l[1] - MATCH_PAD <= cx <= l[3] + MATCH_PAD and l[2] - MATCH_PAD <= cy <= l[4] + MATCH_PAD]
        if hits:
            best = min(hits, key=lambda l: ((l[1] + l[3]) / 2 - cx) ** 2 + ((l[2] + l[4]) / 2 - cy) ** 2)
            free.remove(best)
            tp.append(d)
            found.add(best[0])
        else:
            fp.append(d)
    return tp, fp, found


def pr(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p + r and p == p and r == r else float("nan")
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--labels")
    ap.add_argument("--acne04v2-test", action="store_true")
    ap.add_argument("--weights", default=str(HERE / "skinalyzer2.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--swap-red-blue", action="store_true", help="feed RGB as-is, like a service passing a numpy RGB array")
    ap.add_argument("--draw", action="store_true", help="save annotated images")
    ap.add_argument("--out", help="output folder (default runs/eval/<name>)")
    args = ap.parse_args()

    if args.acne04v2_test:
        img_dir, labels = acne04v2_test_labels()
        set_name = "acne04v2_test"
    else:
        img_dir, labels = Path(args.images), read_labels(args.labels)
        set_name = img_dir.name
    tag = f"{Path(args.weights).stem}_conf{args.conf}_imgsz{args.imgsz}" + ("_swapRB" if args.swap_red_blue else "")
    out = Path(args.out) if args.out else HERE / "runs/eval" / set_name / tag
    out.mkdir(parents=True, exist_ok=True)

    pipe = AppPipeline(args.weights, args.conf, args.imgsz, swap_red_blue=args.swap_red_blue)
    per_view = defaultdict(lambda: [0, 0, 0])      # view -> [tp, fp, fn]
    lesion_views = defaultdict(lambda: [0, 0])     # lesion_id -> [views labelled, views found]
    rows, times, no_face = [], [], []

    for i, (name, lab) in enumerate(sorted(labels.items()), 1):
        res = pipe.run(img_dir / name)
        times.append(res.ms)
        if not res.face_found:
            no_face.append(name)
        tp, fp, found = match(res.boxes, lab["lesions"])
        fn = len(lab["lesions"]) - len(found)
        v = per_view[lab["view"]]
        v[0] += len(tp); v[1] += len(fp); v[2] += fn
        for lid, *_ in lab["lesions"]:
            lesion_views[lid][0] += 1
            lesion_views[lid][1] += lid in found
        rows.append({"image": name, "view": lab["view"], "labels": len(lab["lesions"]), "detections": len(res.boxes),
                     "tp": len(tp), "fp": len(fp), "fn": fn, "face_found": res.face_found, "ms": round(res.ms)})
        if args.draw:
            import cv2
            img = cv2.cvtColor(load_rgb(img_dir / name), cv2.COLOR_RGB2BGR)
            t = max(2, res.width // 600)
            cv2.rectangle(img, res.face_box[:2], res.face_box[2:], (255, 0, 0), t)
            for lid, x1, y1, x2, y2 in lab["lesions"]:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255) if lid not in found else (0, 200, 0), t)
            for d in fp:
                cv2.rectangle(img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 0, 255), t)
            cv2.imwrite(str(out / name), img)
        if i % 25 == 0:
            print(f"  {i}/{len(labels)} images")

    # ----- report -----
    lines = [f"Evaluation: {set_name}", f"Weights: {args.weights}", f"conf {args.conf}, imgsz {args.imgsz}, "
             f"colours {'red/blue swapped (numpy RGB as-is)' if args.swap_red_blue else 'correct'}, CPU, app crop",
             f"Images: {len(labels)}  (no face found in {len(no_face)})",
             f"Speed: {sum(times) / len(times):.0f} ms per image on average (crop + predict, CPU)", "",
             "1. Precision / recall  (a hit = detection centre inside a labelled pimple)",
             f"   {'group':10s} {'labels':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'precision':>10s} {'recall':>8s} {'F1':>6s}"]

    def add(name, tp, fp, fn):
        p, r, f1 = pr(tp, fp, fn)
        lines.append(f"   {name:10s} {tp + fn:7d} {tp:6d} {fp:6d} {fn:6d} {p:10.3f} {r:8.3f} {f1:6.3f}")

    groups = {"frontal": [k for k in per_view if k == "frontal"], "oblique": [k for k in per_view if k in OBLIQUE]}
    for g, keys in groups.items():
        if keys:
            add(g, *(sum(per_view[k][j] for k in keys) for j in range(3)))
    for k in sorted(per_view):
        add(k, *per_view[k])
    add("ALL", *(sum(v[j] for v in per_view.values()) for j in range(3)))

    multi = {lid: v for lid, v in lesion_views.items() if v[0] >= 2}
    if multi:
        lines += ["", "2. Repeatability  (pimples labelled in 2+ views)",
                  f"   {'lesion':12s} {'visible in':>10s} {'found in':>9s} {'share':>6s}  app decision"]
        kept = inconsistent = missed = 0
        for lid, (seen, hit) in sorted(multi.items()):
            if hit == 0:
                decision, missed = "not detected", missed + 1
            elif seen >= 4 and hit < seen / 3:
                decision, inconsistent = "INCONSISTENT (dropped)", inconsistent + 1
            else:
                decision, kept = "counted", kept + 1
            lines.append(f"   {lid:12s} {seen:10d} {hit:9d} {hit / seen:6.0%}  {decision}")
        total_seen = sum(v[0] for v in multi.values())
        total_hit = sum(v[1] for v in multi.values())
        lines += ["", f"   Pimples: {len(multi)}  counted {kept}, inconsistent {inconsistent}, never detected {missed}",
                  f"   Found in {total_hit} of {total_seen} visible views overall ({total_hit / total_seen:.0%})"]
    else:
        lines += ["", "2. Repeatability: no pimple is labelled in 2+ views, so skipped."]
    if no_face:
        lines += ["", "No face found (whole image used): " + ", ".join(no_face)]

    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    with open(out / "per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(report)
    print(f"\nSaved: {out}/report.txt, per_image.csv" + (" and annotated images" if args.draw else ""))


if __name__ == "__main__":
    main()
