"""
Live oiliness (shine) analysis for Skinalyzer.

Oily skin reflects light like a mirror in small patches ("specular
highlights"). Those patches are:
  - much BRIGHTER than the skin around them, and
  - LESS SATURATED (whiter), because the reflected light is the color of the
    lamp, not the color of the skin.

So we look in HSV for pixels that are brighter AND whiter than their local
surroundings. Comparing to the surroundings (not the whole face) means one
side of the face being lit more doesn't count as oil.
"""

import cv2
import numpy as np

from skin_redness import RednessAnalyzer

# Nose bridge down to the tip and the sides of the nose (part of the T-zone)
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 48, 278, 115, 344,
        220, 440, 45, 275]


class OilinessAnalyzer:
    def __init__(self, brightness=15.0, sat_drop=0.75, smoothing=0.15):
        """
        brightness: how much brighter (V, 0-255) than surrounding skin a pixel
                    must be. ~10 sensitive, ~15 moderate, ~25 only strong shine.
        sat_drop:   pixel saturation must be below this fraction of the
                    surrounding saturation (0.75 = at least 25% whiter).
        smoothing:  0-1. Lower = steadier numbers, slower to react.
        """
        self.brightness = brightness
        self.sat_drop = sat_drop
        self.alpha = smoothing
        self._smoothed = {}

    def _ema(self, key, value):
        prev = self._smoothed.get(key)
        new = value if prev is None else prev + self.alpha * (value - prev)
        self._smoothed[key] = new
        return new

    @staticmethod
    def _masked_blur(channel, m, sigma):
        """Average of nearby SKIN pixels only (ignores eyes, background, etc.)."""
        return (cv2.GaussianBlur(channel * m, (0, 0), sigma) /
                (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6))

    def analyze(self, frame, landmarks, regions, acne_boxes=()):
        """
        frame:      raw BGR frame (NOT the one with YOLO boxes drawn on it)
        landmarks:  latest_face_result.face_landmarks[0]
        regions:    {"name": (x1, y1, x2, y2), ...}; a "forehead" entry is
                    combined with the nose to form the T-zone
        acne_boxes: YOLO boxes (whiteheads can look shiny, so excluded)

        Returns (results, shine_mask) or None if not enough skin is visible.
        Each results entry has:
          "shine_pct": % of skin that is shiny
          "intensity": how much brighter the shiny spots are, on average
          "score":     0-100 (placeholder formula, tune it)
        """
        h, w = frame.shape[:2]

        # ---- 1. Skin mask (eyes, brows, lips/teeth already removed)
        mask = RednessAnalyzer.skin_mask(landmarks, h, w)
        for (x1, y1, x2, y2) in acne_boxes:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
        skin = mask > 0
        if skin.sum() < 500:
            return None

        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.int32)
        face_w = int(pts[:, 0].max() - pts[:, 0].min())

        # ---- 2. Brightness (V) and saturation (S)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        S, V = hsv[:, :, 1], hsv[:, :, 2]

        # ---- 3. Compare to surrounding skin (removes gradual lighting changes)
        m = skin.astype(np.float32)
        sigma = max(8.0, 0.10 * face_w)
        bg_V = self._masked_blur(V, m, sigma)
        bg_S = self._masked_blur(S, m, sigma)

        brighter = V - bg_V
        shine = (skin
                 & (brighter > self.brightness)
                 & (S < bg_S * self.sat_drop))
        # Pixels blown out to pure white are shine too (camera maxed out)
        shine |= skin & (V >= 250) & (S < bg_S)

        shine = shine.astype(np.uint8) * 255
        shine = cv2.morphologyEx(shine, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        shine_mask = shine > 0

        # ---- 4. T-zone = forehead box + nose
        t_zone = np.zeros((h, w), np.uint8)
        cv2.fillPoly(t_zone, [cv2.convexHull(pts[NOSE])], 255)
        if "forehead" in regions:
            x1, y1, x2, y2 = regions["forehead"]
            cv2.rectangle(t_zone, (x1, y1), (x2, y2), 255, -1)
        t_zone = (t_zone > 0) & skin

        # ---- 5. Numbers per area
        areas = {"face": skin, "t_zone": t_zone}
        for name, (x1, y1, x2, y2) in regions.items():
            box = np.zeros((h, w), bool)
            box[y1:y2, x1:x2] = True
            areas[name] = box & skin

        results = {}
        for name, area in areas.items():
            if area.sum() < 50:
                continue
            shiny_here = shine_mask & area
            shine_pct = 100.0 * float(shiny_here.sum()) / float(area.sum())
            intensity = float(brighter[shiny_here].mean()) if shiny_here.any() else 0.0
            score = min(100.0, shine_pct * 8)  # tune the 8
            results[name] = {
                "shine_pct": self._ema(name + "_pct", shine_pct),
                "intensity": self._ema(name + "_int", intensity),
                "score": self._ema(name + "_score", score),
            }

        return results, shine_mask

    @staticmethod
    def draw_shine(display, shine_mask, color=(255, 255, 0), strength=0.6):
        """Tint shiny pixels (cyan by default)."""
        tint = np.zeros_like(display)
        tint[:] = color
        blend = cv2.addWeighted(display, 1 - strength, tint, strength, 0)
        display[shine_mask] = blend[shine_mask]
        return display
