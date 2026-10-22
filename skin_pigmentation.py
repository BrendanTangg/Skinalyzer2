"""
Live pigmentation analysis for Skinalyzer.

Finds dark spots and uneven tone using the CIELAB L* (lightness) channel.

Key idea: compare each pixel to the LOCAL average lightness around it, not to
the whole face. Lighting changes slowly across the face (one side brighter,
a soft shadow), but a dark spot is a small, sudden dip. Subtracting a
heavily blurred "background" removes the slow lighting changes and leaves
the spots.
"""

import cv2
import numpy as np

from skin_redness import RednessAnalyzer

# Lower nose / nostrils: the shadows here look exactly like dark spots.
NOSE_LOWER = [48, 115, 220, 45, 4, 275, 440, 344, 278, 294, 327, 326, 2, 97,
              98, 64]


class PigmentationAnalyzer:
    def __init__(self, darkness=5.0, smoothing=0.15):
        """
        darkness:  how many L* units (0-100 scale) darker than the surrounding
                   skin a pixel must be to count as a spot.
                   ~4 sensitive, ~5 moderate, ~8 only obvious spots.
        smoothing: 0-1. Lower = steadier numbers, slower to react.
        """
        self.darkness = darkness
        self.alpha = smoothing
        self._smoothed = {}

    def _ema(self, key, value):
        prev = self._smoothed.get(key)
        new = value if prev is None else prev + self.alpha * (value - prev)
        self._smoothed[key] = new
        return new

    def analyze(self, frame, landmarks, regions, acne_boxes=()):
        """
        frame:      raw BGR frame (NOT the one with YOLO boxes drawn on it)
        landmarks:  latest_face_result.face_landmarks[0]
        regions:    {"name": (x1, y1, x2, y2), ...}
        acne_boxes: YOLO boxes, excluded so acne isn't counted as a dark spot

        Returns (results, spot_mask) or None if not enough skin is visible.
        Each results entry has:
          "spots":    number of dark spots
          "spot_pct": % of skin covered by spots
          "uneven":   tone unevenness (std of local lightness, L* units)
          "score":    0-100 combined score (placeholder formula, tune it)
        """
        h, w = frame.shape[:2]

        # ---- 1. Skin mask: reuse redness mask, also remove nostrils + acne
        mask = RednessAnalyzer.skin_mask(landmarks, h, w)
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.int32)
        face_w = int(pts[:, 0].max() - pts[:, 0].min())

        nose = np.zeros_like(mask)
        cv2.fillPoly(nose, [cv2.convexHull(pts[NOSE_LOWER])], 255)
        k = max(3, int(0.05 * face_w)) | 1
        nose = cv2.dilate(nose, np.ones((k, k), np.uint8))
        mask[nose > 0] = 0

        # Extra erosion at the face edge: hairline and jaw shadows are dark too
        e = max(3, int(0.03 * face_w)) | 1
        mask = cv2.erode(mask, np.ones((e, e), np.uint8))

        for (x1, y1, x2, y2) in acne_boxes:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)

        skin = mask > 0
        skin_px = int(skin.sum())
        if skin_px < 500:
            return None

        # ---- 2. Lightness in real L* units (OpenCV stores L* scaled to 0-255)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
        L = cv2.GaussianBlur(L, (0, 0), 1)  # remove camera noise

        # ---- 3. Local background via a "masked blur"
        # A normal blur would mix in the dark eyes/background near the mask
        # edges. Blurring (L * mask) and dividing by blur(mask) averages only
        # skin pixels.
        m = skin.astype(np.float32)
        sigma = max(5.0, 0.06 * face_w)
        background = (cv2.GaussianBlur(L * m, (0, 0), sigma) /
                      (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6))
        deviation = np.where(skin, L - background, 0)  # negative = darker

        # ---- 4. Spots: pixels clearly darker than their surroundings
        spots = ((-deviation > self.darkness) & skin).astype(np.uint8) * 255
        spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Keep blobs within a sensible size. Tiny = noise/pores,
        # huge = a shadow, not a spot.
        n, labels, stats, cents = cv2.connectedComponentsWithStats(spots, 8)
        min_area = max(4, int(skin_px * 0.00005))
        max_area = int(skin_px * 0.02)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
                keep[i] = True
        spot_mask = keep[labels]
        spot_centers = cents[keep]

        # ---- 5. Numbers per region
        results = {}
        all_regions = {"face": (0, 0, w, h), **regions}
        for name, (x1, y1, x2, y2) in all_regions.items():
            roi_skin = skin[y1:y2, x1:x2]
            if roi_skin.sum() < 50:
                continue
            dev = deviation[y1:y2, x1:x2][roi_skin]
            spot_pct = 100.0 * float(spot_mask[y1:y2, x1:x2][roi_skin].mean())
            uneven = float(dev.std())
            count = int(sum(1 for (cx, cy) in spot_centers
                            if x1 <= cx < x2 and y1 <= cy < y2))
            score = min(100.0, spot_pct * 10 + uneven * 10)  # tune these weights
            results[name] = {
                "spots": self._ema(name + "_spots", count),
                "spot_pct": self._ema(name + "_pct", spot_pct),
                "uneven": self._ema(name + "_uneven", uneven),
                "score": self._ema(name + "_score", score),
            }

        return results, spot_mask

    @staticmethod
    def draw_spots(display, spot_mask, color=(255, 0, 255)):
        """Outline detected spots (magenta by default)."""
        contours, _ = cv2.findContours(spot_mask.astype(np.uint8) * 255,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, color, 1)
        return display
