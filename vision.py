import cv2
import numpy as np

# Define standard cube colors (BGR values approx.)
CUBE_COLORS = {
    "W": (255, 255, 255),  # White
    "Y": (0, 255, 255),    # Yellow
    "R": (0, 0, 255),      # Red
    "O": (0, 165, 255),    # Orange
    "B": (255, 0, 0),      # Blue
    "G": (0, 255, 0)       # Green
}

def detect_face_colors(image_path):
    """Detects 9 sticker colors from a cube face image"""
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # divide into 3x3 grid
    step_h, step_w = h // 3, w // 3
    colors = []

    for row in range(3):
        for col in range(3):
            y1, y2 = row * step_h, (row + 1) * step_h
            x1, x2 = col * step_w, (col + 1) * step_w

            patch = img[y1:y2, x1:x2]
            avg_color = np.mean(patch, axis=(0, 1))

            # find closest cube color
            detected = min(CUBE_COLORS, key=lambda c: np.linalg.norm(avg_color - CUBE_COLORS[c]))
            colors.append(detected)

    return "".join(colors)  # return like "WGRBYO..."
