"""
Rubik's Cube from-photos solver (improved color detection + human-readable move instructions)
Dependencies: pip install opencv-python numpy scikit-learn kociemba
Usage:
    - Prepare six photos and name them clearly: up.jpg, right.jpg, front.jpg, down.jpg, left.jpg, back.jpg
      OR provide a dict mapping face labels ('U','R','F','D','L','B') to file paths.
    - Hold the cube while taking photos: WHITE on UP, GREEN on FRONT.
    - Run: python rubik_solver_photos.py
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import kociemba
import os
import math

# -------------- utils for perspective warp / ordering points ----------------
def order_points(pts):
    # expects pts as array of 4 points
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def four_point_transform(image, pts, size=300):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB, size))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB, size))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.resize(warped, (size, size))
    return warped

# -------------- face extraction & cell averaging ----------------
def extract_face_cells(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not open {img_path}")
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quad = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4,2)
            break

    if quad is not None:
        warped = four_point_transform(orig, quad, size=300)
    else:
        # fallback: center crop square
        h, w = orig.shape[:2]
        s = min(h, w)
        cx, cy = w//2, h//2
        cropped = orig[cy - s//2: cy + s//2, cx - s//2: cx + s//2]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            warped = cv2.resize(orig, (300,300))
        else:
            warped = cv2.resize(cropped, (300,300))

    # break into 3x3 grid and compute mean color (in Lab)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    cells = []
    cell_size = warped.shape[0] // 3
    for r in range(3):
        row = []
        for c in range(3):
            y1 = r * cell_size + cell_size//8
            y2 = (r+1) * cell_size - cell_size//8
            x1 = c * cell_size + cell_size//8
            x2 = (c+1) * cell_size - cell_size//8
            patch = lab[y1:y2, x1:x2]
            mean = np.mean(patch.reshape(-1, 3), axis=0)
            row.append(mean)   # Lab mean
        cells.append(row)
    if debug:
        cv2.imwrite("debug_warped_" + os.path.basename(img_path), warped)
    return np.array(cells)  # shape (3,3,3) Lab means

# -------------- main processing ----------------
FACES_ORDER = ['U','R','F','D','L','B']

def auto_label_from_filename(path):
    name = os.path.basename(path).lower()
    if any(x in name for x in ['up','u','top']):
        return 'U'
    if any(x in name for x in ['down','d','bottom','dn']):
        return 'D'
    if any(x in name for x in ['front','f','fr']):
        return 'F'
    if any(x in name for x in ['back','b','rear','behind']):
        return 'B'
    if any(x in name for x in ['left','l']):
        return 'L'
    if any(x in name for x in ['right','r']):
        return 'R'
    return None

def build_cube_state_from_images(face_image_paths, debug=False):
    """
    face_image_paths: either:
       - dict with keys 'U','R','F','D','L','B' -> file paths
       OR
       - list of 6 file paths (will try to auto-detect label from filename)
    returns: cube_state_string (54 chars), mapping info
    """
    # normalize input
    if isinstance(face_image_paths, list):
        # try to auto assign
        mapping = {}
        for p in face_image_paths:
            label = auto_label_from_filename(p)
            if label is None:
                raise ValueError(f"Filename '{p}' must contain face hint (up/down/front/back/left/right) or supply a dict with labels.")
            mapping[label] = p
        face_image_paths = mapping

    # ensure all faces present
    for f in FACES_ORDER:
        if f not in face_image_paths:
            raise ValueError(f"Missing image for face {f}. Provide file for '{f}'. Filenames should include face or pass a dict mapping.")

    # Extract cells
    face_cells_lab = {}
    for face in FACES_ORDER:
        face_cells_lab[face] = extract_face_cells(face_image_paths[face], debug=debug)  # shape 3x3x3

    # Gather center colors
    center_colors = {}
    for face in FACES_ORDER:
        center_colors[face] = face_cells_lab[face][1,1]  # Lab mean

    # Check uniqueness of centers (if too close, we'll fallback to KMeans on all stickers)
    centers_arr = np.array(list(center_colors.values()))
    dists = np.linalg.norm(centers_arr[:,None,:] - centers_arr[None,:,:], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min()
    # threshold: if two centers closer than 12 (Lab units), treat ambiguous
    ambiguous = min_pair < 12

    # Build mapping from sticker -> face letter by nearest center or via KMeans
    stickers_lab = []
    sticker_face_ref = []  # (face, r, c)
    for face in FACES_ORDER:
        for r in range(3):
            for c in range(3):
                stickers_lab.append(face_cells_lab[face][r,c])
                sticker_face_ref.append((face,r,c))
    stickers_arr = np.array(stickers_lab)

    if not ambiguous:
        # simple nearest-center assignment
        face_centers = {face: center_colors[face] for face in FACES_ORDER}
        face_center_arr = np.array([face_centers[f] for f in FACES_ORDER])
        labels = []
        for s in stickers_arr:
            # distance to each center
            d = np.linalg.norm(face_center_arr - s, axis=1)
            labels.append(FACES_ORDER[int(np.argmin(d))])
    else:
        # fallback: cluster all stickers into 6 clusters (KMeans) in Lab space
        kmeans = KMeans(n_clusters=6, random_state=0).fit(stickers_arr)
        cluster_centers = kmeans.cluster_centers_
        sticker_cluster = kmeans.labels_
        # determine which cluster corresponds to each face by using the center cell of the provided face images
        cluster_to_face = {}
        for face_idx, face in enumerate(FACES_ORDER):
            center = center_colors[face]
            # find closest cluster center to this center
            d = np.linalg.norm(cluster_centers - center, axis=1)
            chosen_cluster = int(np.argmin(d))
            if chosen_cluster in cluster_to_face and cluster_to_face[chosen_cluster] != face:
                # conflict: choose next best cluster
                sorted_idx = np.argsort(d)
                for idx in sorted_idx:
                    if idx not in cluster_to_face:
                        chosen_cluster = int(idx)
                        break
            cluster_to_face[chosen_cluster] = face
        # now map each sticker to face label via its cluster
        labels = []
        for cl in sticker_cluster:
            if cl in cluster_to_face:
                labels.append(cluster_to_face[cl])
            else:
                # fallback: nearest cluster center mapping (shouldn't happen)
                labels.append(cluster_to_face[list(cluster_to_face.keys())[0]])
    
    # Build cube state in kociemba order: U, R, F, D, L, B each 9 stickers row-major (toprow left->right)
    face_to_labels = {f: [] for f in FACES_ORDER}
    for (face,r,c), lab, assigned in zip(sticker_face_ref, stickers_arr, labels):
        # assigned is the face letter which this sticker is identified as (U/R/F/D/L/B)
        # For kociemba we need each face's 9 stickers in the correct order, so we append to the original face bucket:
        face_to_labels[face].append(assigned)

    # Convert face_to_labels[face] (which are assignments for each sticker in that photographed face)
    # to a 9-letter string where each letter is the face-letter of that sticker (i.e., color-coded)
    cube_str_parts = []
    for f in FACES_ORDER:
        # face_to_labels[f] currently has 9 items added in the order we iterated (r,c)
        # ensure length 9
        arr = face_to_labels[f]
        if len(arr) != 9:
            raise RuntimeError("Internal error building stickers")
        cube_str_parts.append(''.join(arr))

    cube_state = ''.join(cube_str_parts)
    # sanity check: cube_state should contain exactly 6 unique letters: U,R,F,D,L,B each repeated 9 times
    uniq, counts = np.unique(list(cube_state), return_counts=True)
    # if counts are not 9 each, something's off
    # we will try to fix by simple re-labelling using center_colors: if some letter missing, assign cluster-based mapping
    expected_set = set(FACES_ORDER)
    found_set = set(uniq)
    if found_set != expected_set:
        # try robust fix: map each sticker to nearest of the *center color vectors* (centers_arr)
        corrected = []
        for s in stickers_arr:
            d = np.linalg.norm(centers_arr - s, axis=1)
            idx = int(np.argmin(d))
            corrected.append(FACES_ORDER[idx])
        # rebuild per photographed face order:
        idx = 0
        face_to_labels_corr = {f: [] for f in FACES_ORDER}
        for face in FACES_ORDER:
            for _ in range(9):
                face_to_labels_corr[face].append(corrected[idx]); idx+=1
        cube_state = ''.join(''.join(face_to_labels_corr[f]) for f in FACES_ORDER)
        uniq, counts = np.unique(list(cube_state), return_counts=True)
        if set(uniq) != expected_set:
            raise RuntimeError(f"Could not reliably map colors to faces. Found colors: {uniq}. Please retake photos with uniform lighting and ensure WHITE on UP and GREEN on FRONT.")

    # final sanity: each face-letter should appear 9 times
    for f in FACES_ORDER:
        if cube_state.count(f) != 9:
            raise RuntimeError(f"Face {f} appears {cube_state.count(f)} times in generated cube state (expected 9).")

    return cube_state, {
        'face_cells_lab': face_cells_lab,
        'center_colors': center_colors,
        'ambiguous_centers': ambiguous
    }

# -------------- translate moves into row/column instructions ----------------
MOVE_DESCRIPTIONS = {
    'U': {
        'face_name': 'Up',
        'layer': 'top row (row 1)',
        'how_to_view': 'look down at the Up face (from above)',
        'physical': 'Rotate the top layer 90° clockwise when looking at the Up face from above.'
    },
    'D': {
        'face_name': 'Down',
        'layer': 'bottom row (row 3)',
        'how_to_view': 'look at the Down face (from below)',
        'physical': 'Rotate the bottom layer 90° clockwise when looking at the Down face.'
    },
    'F': {
        'face_name': 'Front',
        'layer': 'front face (the whole face)',
        'how_to_view': 'look directly at the Front face',
        'physical': 'Rotate the front face 90° clockwise when looking at the front face (this rotates the top row, right column, bottom row and left column of adjacent faces accordingly).'
    },
    'B': {
        'face_name': 'Back',
        'layer': 'back face (the whole face)',
        'how_to_view': 'look directly at the Back face',
        'physical': 'Rotate the back face 90° clockwise when looking at the back face.'
    },
    'L': {
        'face_name': 'Left',
        'layer': 'leftmost column (column 1 when looking at Front)',
        'how_to_view': 'look directly at the Left face',
        'physical': 'Rotate the left face 90° clockwise when looking at the left face (equivalently rotate the leftmost column).'
    },
    'R': {
        'face_name': 'Right',
        'layer': 'rightmost column (column 3 when looking at Front)',
        'how_to_view': 'look directly at the Right face',
        'physical': 'Rotate the right face 90° clockwise when looking at the right face (equivalently rotate the rightmost column).'
    },
}

def describe_move(move):
    """
    move: string like 'R', "R'", 'R2'
    returns: human readable instruction
    """
    base = move[0]
    modifier = move[1:]  # '' or "'" or '2'
    desc = MOVE_DESCRIPTIONS.get(base, None)
    if desc is None:
        return f"Unknown move {move}"

    if modifier == '':
        angle = "90° clockwise"
        extra = "one quarter turn clockwise"
    elif modifier == "'":
        angle = "90° counter-clockwise"
        extra = "one quarter turn counter-clockwise"
    elif modifier == '2':
        angle = "180°"
        extra = "half turn (180°)"
    else:
        angle = modifier
        extra = f"{modifier}"

    text = (
        f"{base}{modifier}: Turn the {desc['face_name']} face {angle} ({extra}).\n"
        f" - Layer: {desc['layer']} (relative to Front view).\n"
        f" - How to view: {desc['how_to_view']}.\n"
        f" - Physical instruction: {desc['physical']}\n"
    )
    # add quick tip mapping to front-view rows/cols
    if base in ['L','R']:
        text += f"   → Equivalent: rotate the {'rightmost' if base=='R' else 'leftmost'} column (column {'3' if base=='R' else '1'}) when you hold WHITE up and GREEN front.\n"
    if base in ['U','D']:
        text += f"   → Equivalent: rotate the {'top' if base=='U' else 'bottom'} horizontal layer (row {'1' if base=='U' else '3'}) when you hold WHITE up and GREEN front.\n"
    if base in ['F','B']:
        text += f"   → Equivalent: rotate the entire front/back face (affects surrounding rows/columns of adjacent faces).\n"

    return text

# -------------- top-level solve function ----------------
def solve_from_images(face_image_paths, debug=False):
    # Build cube state
    cube_state, meta = build_cube_state_from_images(face_image_paths, debug=debug)
    # Now pass to kociemba
    try:
        solution = kociemba.solve(cube_state)
    except Exception as e:
        raise RuntimeError(f"kociemba failed to solve. Error: {e}\nCube state: {cube_state}")

    moves = solution.split()
    # produce detailed instructions
    instructions = []
    for i, m in enumerate(moves, start=1):
        instructions.append({
            'step': i,
            'move': m,
            'instruction_text': describe_move(m)
        })

    return {
        'cube_state': cube_state,
        'solution': solution,
        'moves': moves,
        'instructions': instructions,
        'meta': meta
    }

# ------------------- Example usage -------------------
if __name__ == "__main__":
    # Provide your images here. It's strongly recommended filenames contain face names
    # OR provide a dict mapping 'U','R','F','D','L','B' to file paths.
    face_images = {
        'U': 'up.jpg',
        'R': 'right.jpg',
        'F': 'front.jpg',
        'D': 'down.jpg',
        'L': 'left.jpg',
        'B': 'back.jpg'
    }

    # Run solver
    try:
        res = solve_from_images(face_images, debug=False)
        print("Detected cube state (54 chars):", res['cube_state'])
        print("Solution (kociemba):", res['solution'])
        print("\nStep-by-step with row/column guidance:")
        for s in res['instructions']:
            print(f"\nStep {s['step']}: {s['move']}\n{s['instruction_text']}")
    except Exception as e:
        print("ERROR:", e)
