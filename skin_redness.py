"""
Live redness analysis for Skinalyzer.

Works with MediaPipe Tasks FaceLandmarker landmarks (normalized x, y).
Redness = how much higher a pixel's CIELAB a* (green <-> red) is than the
median a* of this person's own face skin. Comparing to their own baseline
makes it work across skin tones.
"""

import cv2
import numpy as np

# MediaPipe Face Mesh landmark indices (ordered polygons)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
            144, 163, 7]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374,
             373, 390, 249]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
        17, 84, 181, 91, 146]
# Lips are naturally red, so they must be excluded or they skew everything.
EXCLUDE = [LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS]


class RednessAnalyzer:
    def __init__(self, threshold=8.0, smoothing=0.15):
        """
        threshold: a* units above the face baseline that count as "red".
                   ~6 sensitive, ~8 moderate, ~12 only strong redness.
        smoothing: 0-1. Lower = steadier numbers, slower to react.
        """
        self.threshold = threshold
        self.alpha = smoothing
        self._smoothed = {}
        self._wb_gains = None

    # ---------- helpers ----------
    def _ema(self, key, value):
        prev = self._smoothed.get(key)
        new = value if prev is None else prev + self.alpha * (value - prev)
        self._smoothed[key] = new
        return new

    def _white_balance(self, frame):
        """Gray-world white balance, smoothed over time so it doesn't flicker."""
        means = frame.reshape(-1, 3).mean(axis=0)
        gains = means.mean() / (means + 1e-6)
        if self._wb_gains is None:
            self._wb_gains = gains
        else:
            self._wb_gains += self.alpha * (gains - self._wb_gains)
        out = frame.astype(np.float32) * self._wb_gains
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def skin_mask(landmarks, h, w):
        """Face skin mask from landmarks, with eyes, brows and lips removed."""
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts[FACE_OVAL]], 255)

        excl = np.zeros_like(mask)
        for region in EXCLUDE:
            cv2.fillPoly(excl, [cv2.convexHull(pts[region])], 255)
        face_w = pts[:, 0].max() - pts[:, 0].min()
        k = max(3, int(0.04 * face_w)) | 1
        excl = cv2.dilate(excl, np.ones((k, k), np.uint8))

        mask[excl > 0] = 0
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        return mask

    # ---------- main entry ----------
    def analyze(self, frame, landmarks, regions, acne_boxes=()):
        """
        frame:      raw BGR frame (NOT the one with YOLO boxes drawn on it)
        landmarks:  latest_face_result.face_landmarks[0]
        regions:    {"name": (x1, y1, x2, y2), ...} e.g. cheeks and forehead
        acne_boxes: iterable of (x1, y1, x2, y2) from YOLO, excluded so acne
                    doesn't count as general redness

        Returns (results, excess_map, skin) or None if not enough skin is visible.
        results = {"face": {...}, "left_cheek": {...}, ...} where each entry has
                  "score" (0-100) and "red_pct" (% of skin pixels above threshold).
        """
        h, w = frame.shape[:2]
        img = self._white_balance(frame)

        mask = self.skin_mask(landmarks, h, w)
        for (x1, y1, x2, y2) in acne_boxes:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
        skin = mask > 0
        if skin.sum() < 500:
            return None

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32) - 128.0   # OpenCV offsets a* by 128
        a = cv2.GaussianBlur(a, (0, 0), 2)

        baseline = float(np.median(a[skin]))
        excess = np.clip(a - baseline, 0, None)

        results = {}
        all_regions = {"face": (0, 0, w, h), **regions}
        for name, (x1, y1, x2, y2) in all_regions.items():
            roi_skin = skin[y1:y2, x1:x2]
            vals = excess[y1:y2, x1:x2][roi_skin]
            if vals.size < 50:
                continue
            score = min(100.0, float(vals.mean()) * 10)          # tune the 10
            red_pct = 100.0 * float((vals > self.threshold).mean())
            results[name] = {
                "score": self._ema(name + "_score", score),
                "red_pct": self._ema(name + "_pct", red_pct),
            }

        # Absolute baseline: catches whole-face flushing, which the
        # relative score can't see (if everything is red, nothing stands out).
        results["face"]["baseline_a"] = self._ema("baseline", baseline)
        return results, excess, skin

    @staticmethod
    def draw_heatmap(display, excess, skin, strength=0.45):
        heat = cv2.applyColorMap(np.clip(excess * 15, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        blend = cv2.addWeighted(display, 1 - strength, heat, strength, 0)
        display[skin] = blend[skin]
        return display
