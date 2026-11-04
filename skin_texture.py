"""
Live texture analysis for Skinalyzer.

Texture = how uneven the skin SURFACE is, split into two parts:
  - lines:     long, thin structures running one way (fine lines, wrinkles)
  - roughness: bumpy unevenness with no main direction

Only brightness (CIELAB L*) is used, after removing slow lighting changes,
so colour (redness, pigment) mostly doesn't count. A bank of Gabor filters
at several directions tells the two parts apart: a line responds strongly to
one direction, a bump responds to all of them about equally.

Pores, acne, shine and dark spots are masked out first. Otherwise texture
would just be measuring them again under a different name.

The face is resized to a fixed width before filtering, so the filter sizes
mean the same thing whether you sit close to or far from the camera.
"""

import cv2
import numpy as np

from skin_redness import RednessAnalyzer
from skin_pigmentation import NOSE_LOWER
from skin_oiliness import NOSE


class TextureAnalyzer:
    def __init__(self, face_width=320, wavelengths=(4, 8, 16), n_orient=8,
                 pore_contrast=0.04, max_contrast=0.08, smoothing=0.15):
        """
        face_width:    the face is resized to this many pixels wide first.
        wavelengths:   filter sizes in pixels at face_width. Small = fine lines
                       and fine roughness, large = deeper creases.
        n_orient:      number of filter directions.
        pore_contrast: how much darker (log brightness) than nearby skin a tiny
                       blob must be to be masked out as a pore. Lower = masks more.
        max_contrast:  detail is clipped to +/- this (log brightness, 0.08 = ~8%)
                       so hard shadows and lashes can't drown out skin texture.
        smoothing:     0-1. Lower = steadier numbers, slower to react.
        """
        self.face_width = face_width
        self.wavelengths = wavelengths
        self.pore_contrast = pore_contrast
        self.max_contrast = max_contrast
        self.alpha = smoothing
        self._smoothed = {}

        thetas = np.arange(n_orient) * np.pi / n_orient
        # Directions are doubled so that 0 and 180 degrees count as the same line
        self._cos2 = np.cos(2 * thetas)[:, None, None]
        self._sin2 = np.sin(2 * thetas)[:, None, None]
        # Bigger filters run on a shrunk image (same result, much faster):
        # wavelength 8 runs as 4 at half size, 16 as 4 at quarter size, etc.
        self._bank = {}
        for wl in wavelengths:
            shrink = max(1, wl // wavelengths[0])
            self._bank[wl] = (shrink, self._gabor_pairs(wl / shrink, thetas))

    # ---------- helpers ----------
    def _ema(self, key, value):
        prev = self._smoothed.get(key)
        new = value if prev is None else prev + self.alpha * (value - prev)
        self._smoothed[key] = new
        return new

    @staticmethod
    def _gabor_pairs(wavelength, thetas):
        """Even/odd Gabor kernels per direction, scaled so every size has equal weight."""
        sigma = 0.56 * wavelength  # about one octave of bandwidth
        k = int(6 * sigma) | 1
        pairs = []
        for t in thetas:
            even = cv2.getGaborKernel((k, k), sigma, t, wavelength, 0.5, 0, ktype=cv2.CV_32F)
            odd = cv2.getGaborKernel((k, k), sigma, t, wavelength, 0.5, np.pi / 2, ktype=cv2.CV_32F)
            even -= even.mean()  # flat skin must give zero
            norm = np.sqrt((even ** 2).sum()) + 1e-9
            pairs.append((even / norm, odd / norm))
        return pairs

    @staticmethod
    def _masked_blur(channel, m, sigma):
        """Average of nearby SKIN pixels only (ignores eyes, background, etc.)."""
        return (cv2.GaussianBlur(channel * m, (0, 0), sigma) /
                (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6))

    # ---------- main entry ----------
    def analyze(self, frame, landmarks, regions, acne_boxes=(), exclude=None):
        """
        frame:      raw BGR frame (NOT the one with YOLO boxes drawn on it)
        landmarks:  latest_face_result.face_landmarks[0]
        regions:    {"name": (x1, y1, x2, y2), ...} e.g. cheeks and forehead
        acne_boxes: YOLO boxes, excluded so lesions don't count as texture
        exclude:    optional full-frame bool mask of other pixels to skip,
                    e.g. pigment spots | shine

        Returns (results, line_map, skin) or None if not enough skin is visible.
        Each results entry has:
          "lines":  energy of one-direction structures (fine lines, wrinkles)
          "rough":  energy of no-direction unevenness
          "score":  0-100 (placeholder formula, tune it)
        line_map is full-frame, for drawing.
        """
        h, w = frame.shape[:2]

        # ---- 1. Skin mask (eyes, brows, lips removed), minus acne and extras
        mask = RednessAnalyzer.skin_mask(landmarks, h, w)
        for (x1, y1, x2, y2) in acne_boxes:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
        if exclude is not None:
            mask[exclude] = 0

        # The nose's 3D shading and nostrils are hard edges, not texture
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.int32)
        face_w = int(pts[:, 0].max() - pts[:, 0].min())
        nose = np.zeros_like(mask)
        cv2.fillPoly(nose, [cv2.convexHull(pts[NOSE + NOSE_LOWER])], 255)
        k = max(3, int(0.05 * face_w)) | 1
        mask[cv2.dilate(nose, np.ones((k, k), np.uint8)) > 0] = 0
        if (mask > 0).sum() < 500:
            return None

        # ---- 2. Crop to the face and resize to a fixed face width
        fx1, fy1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
        fx2, fy2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
        if fx2 - fx1 < 20 or fy2 - fy1 < 20:
            return None
        scale = self.face_width / float(fx2 - fx1)
        size = (self.face_width, max(1, int(round((fy2 - fy1) * scale))))
        crop = cv2.resize(frame[fy1:fy2, fx1:fx2], size, interpolation=cv2.INTER_AREA)
        skin = cv2.resize(mask[fy1:fy2, fx1:fx2], size, interpolation=cv2.INTER_NEAREST) > 0
        m = skin.astype(np.float32)

        # ---- 3. Detail layer: log brightness minus its slow lighting changes.
        # Log means a brighter or darker exposure cancels out.
        L = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        logL = np.log(L + 1.0)
        detail = logL - self._masked_blur(logL, m, 2.0 * max(self.wavelengths))
        detail = np.clip(detail, -self.max_contrast, self.max_contrast)

        # ---- 4. Pores: tiny blobs darker than their surroundings
        dog = cv2.GaussianBlur(detail, (0, 0), 1.0) - cv2.GaussianBlur(detail, (0, 0), 2.0)
        pores = cv2.dilate(((dog < -self.pore_contrast) & skin).astype(np.uint8),
                           np.ones((3, 3), np.uint8)) > 0
        valid = skin & ~pores
        v = valid.astype(np.float32)

        # Fill masked pixels with nearby skin so their edges don't look like lines
        fill = self._masked_blur(detail, v, 3.0)
        detail = np.where(valid, detail, fill).astype(np.float32)

        # ---- 5. Gabor energy per size: split into one-direction vs no-direction
        line_map = np.zeros_like(detail)
        rough_map = np.zeros_like(detail)
        dh, dw = detail.shape
        for wl, (shrink, pairs) in self._bank.items():
            img = detail
            if shrink > 1:
                img = cv2.resize(detail, (max(1, dw // shrink), max(1, dh // shrink)),
                                 interpolation=cv2.INTER_AREA)
            energy = np.stack([cv2.filter2D(img, cv2.CV_32F, even) ** 2 +
                               cv2.filter2D(img, cv2.CV_32F, odd) ** 2
                               for even, odd in pairs])
            total = energy.sum(axis=0) + 1e-12
            # 0 = same response in every direction, 1 = one direction only
            coherence = np.hypot((energy * self._cos2).sum(axis=0),
                                 (energy * self._sin2).sum(axis=0)) / total
            lines, rough = total * coherence, total * (1.0 - coherence)
            if shrink > 1:
                lines = cv2.resize(lines, (dw, dh), interpolation=cv2.INTER_LINEAR)
                rough = cv2.resize(rough, (dw, dh), interpolation=cv2.INTER_LINEAR)
            line_map += lines
            rough_map += rough

        # Stay away from the skin edge by the reach of the biggest filter, so
        # eyes, hairline and the face outline don't show up as lines. Pores are
        # already filled, so only the outer skin mask is shrunk (shrinking
        # `valid` would turn every pore into a big square hole).
        # (borderValue=0: outside the crop is NOT skin, or the crop edge survives)
        e = 2 * max(self.wavelengths) + 1
        inner = cv2.erode(skin.astype(np.uint8), np.ones((e, e), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        inner &= valid

        # ---- 6. Numbers per area (region boxes converted to crop coordinates)
        results = {}
        all_regions = {"face": (fx1, fy1, fx2, fy2), **regions}
        for name, (x1, y1, x2, y2) in all_regions.items():
            cx1, cy1 = int((x1 - fx1) * scale), int((y1 - fy1) * scale)
            cx2, cy2 = int((x2 - fx1) * scale), int((y2 - fy1) * scale)
            cx1, cy1 = max(0, cx1), max(0, cy1)
            area = inner[cy1:cy2, cx1:cx2]
            if area.sum() < 50:
                continue
            lines = float(line_map[cy1:cy2, cx1:cx2][area].mean()) * 1000
            rough = float(rough_map[cy1:cy2, cx1:cx2][area].mean()) * 1000
            score = min(100.0, (lines + rough) * 1.5)  # tune the 1.5
            results[name] = {
                "lines": self._ema(name + "_lines", lines),
                "rough": self._ema(name + "_rough", rough),
                "score": self._ema(name + "_score", score),
            }

        if "face" not in results:
            return None

        # ---- 7. Line map back to full-frame size for drawing
        full = np.zeros((h, w), np.float32)
        full[fy1:fy2, fx1:fx2] = cv2.resize(np.where(inner, line_map, 0),
                                            (fx2 - fx1, fy2 - fy1),
                                            interpolation=cv2.INTER_LINEAR)
        skin_full = np.zeros((h, w), bool)
        skin_full[fy1:fy2, fx1:fx2] = cv2.resize(inner.astype(np.uint8),
                                                 (fx2 - fx1, fy2 - fy1),
                                                 interpolation=cv2.INTER_NEAREST) > 0
        return results, full, skin_full

    @staticmethod
    def draw_lines(display, line_map, skin, color=(0, 255, 255), strength=0.8):
        """Tint the strongest line-like texture (yellow by default)."""
        vals = line_map[skin]
        if vals.size == 0:
            return display
        top = float(np.percentile(vals, 99)) + 1e-12
        a = np.clip(line_map / top, 0, 1) ** 2 * strength  # squared: show strong lines only
        a[~skin] = 0
        a = a[:, :, None]
        tint = np.empty_like(display)
        tint[:] = color
        display[:] = (display * (1 - a) + tint * a).astype(np.uint8)
        return display
