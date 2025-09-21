# vision.py
import cv2
import numpy as np

# Reference BGR colors (typical sticker colors). We'll convert to LAB for perceptual comparison.
REF_BGR = {
    "W": np.array([255, 255, 255], dtype=np.uint8),  # White
    "Y": np.array([0, 255, 255], dtype=np.uint8),    # Yellow (BGR)
    "R": np.array([0, 0, 255], dtype=np.uint8),      # Red
    "O": np.array([0, 165, 255], dtype=np.uint8),    # Orange
    "B": np.array([255, 0, 0], dtype=np.uint8),      # Blue
    "G": np.array([0, 255, 0], dtype=np.uint8)       # Green
}

# Precompute LAB references
REF_LAB = {}
for k, bgr in REF_BGR.items():
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0].astype(float)
    REF_LAB[k] = lab

def bgr_to_lab(bgr):
    """Convert a 3-element BGR array (uint8) to LAB (float)."""
    lab = cv2.cvtColor(np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]]), cv2.COLOR_BGR2LAB)[0][0].astype(float)
    return lab

def closest_color(bgr):
    """Return the color key (W,Y,R,O,B,G) closest to the provided BGR color."""
    lab = bgr_to_lab(bgr)
    best = None
    best_dist = float("inf")
    for k, ref in REF_LAB.items():
        d = np.linalg.norm(lab - ref)
        if d < best_dist:
            best_dist = d
            best = k
    return best

def detect_face_colors_from_image(img_bgr, debug=False):
    """
    Input:
      img_bgr : OpenCV BGR numpy array representing one face photo
    Returns:
      list of 9 color letters (row-major order) among W,Y,R,O,B,G
    Notes:
      - The function assumes the uploaded photo shows a single cube face reasonably centered.
      - It resizes to square and divides into a 3x3 grid. For each cell it samples the central patch
        (to avoid borders) and chooses the nearest reference color.
    """
    # Make square and resize
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    # crop center square
    cx, cy = w // 2, h // 2
    half = side // 2
    crop = img_bgr[cy - half:cy + half, cx - half:cx + half].copy()
    size = 300
    face = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    stickers = []
    cell = size // 3
    margin = int(cell * 0.22)  # avoid borders, sample center ~56% area
    for r in range(3):
        for c in range(3):
            y1 = r * cell + margin
            y2 = (r + 1) * cell - margin
            x1 = c * cell + margin
            x2 = (c + 1) * cell - margin
            if y2 <= y1 or x2 <= x1:
                patch = face[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            else:
                patch = face[y1:y2, x1:x2]
            avg_bgr = np.mean(patch.reshape(-1, 3), axis=0)
            color = closest_color(avg_bgr[::-1] if False else avg_bgr)  # avg_bgr is BGR already
            stickers.append(color)
            if debug:
                print(f"cell {r},{c} avg {avg_bgr} -> {color}")

    return stickers
