# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from collections import Counter

from vision import detect_face_colors_from_image
from solver import solve_kociemba

st.set_page_config(page_title="Rubik's Cube AI Solver", layout="wide")

# --- CSS styling
st.markdown("""
<style>
body { background: linear-gradient(135deg, #071A2D, #0B3B56); color: #e6f7ff; }
h1 { color:#FFD166; text-align:center; }
.card { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.small { font-size:0.9rem; color:#cfeefb; }
.grid { display:flex; gap:8px; flex-wrap:wrap; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 Rubik's Cube AI Solver — Step by Step</h1>", unsafe_allow_html=True)
st.markdown('<div class="card small">Upload 6 face images (any order). Make sure each image shows one face centered & reasonably straight.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload 6 face images (jpg/png)", accept_multiple_files=True, type=["jpg","jpeg","png"])

# mapping from sticker color letter -> face letter used by kociemba
COLOR_TO_FACE = {"W":"U", "R":"R", "G":"F", "Y":"D", "O":"L", "B":"B"}
# mapping of desired face order to center sticker color
FACE_CENTER = {"U":"W","R":"R","F":"G","D":"Y","L":"O","B":"B"}
face_order = ["U","R","F","D","L","B"]

def decode_uploaded_image(uploaded_file):
    """
    Returns: (img_bgr: np.ndarray, preview_pil: PIL.Image, raw_bytes: bytes)
    Uses getvalue() so the buffer isn't consumed multiple times.
    """
    raw = uploaded_file.getvalue()  # does not consume the file pointer for Streamlit
    arr = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # preview from the same bytes
    preview_pil = Image.open(BytesIO(raw)).convert("RGB")
    return img_bgr, preview_pil, raw

def build_cube_string(detected_faces):
    """
    detected_faces: list of dicts [{'center': 'W', 'stickers': ['W','R',...'], 'preview': PIL.Image}, ...]
    Returns 54-char string or raises ValueError.
    """
    # Map center color -> stickers
    center_map = {d['center']: d['stickers'] for d in detected_faces}

    # Validate we have all 6 centers
    needed_centers = set(FACE_CENTER.values())  # {'W','R','G','Y','O','B'}
    if set(center_map.keys()) != needed_centers:
        raise ValueError(f"Detected centers mismatch. Found centers {sorted(center_map.keys())}, expected {sorted(needed_centers)}.")

    # Build in URFDLB face order
    parts = []
    for face in face_order:
        center_color = FACE_CENTER[face]  # e.g. 'W' for U
        stickers = center_map[center_color]  # list of 9 color letters
        if len(stickers) != 9:
            raise ValueError(f"Face with center {center_color} has {len(stickers)} stickers, expected 9.")
        # Convert color letters (W,Y,R,...) to face letters (U,R,F,...) expected by kociemba
        try:
            mapped = [COLOR_TO_FACE[c] for c in stickers]
        except KeyError as ke:
            raise ValueError(f"Unknown color '{ke.args[0]}' detected. Allowed colors: {sorted(COLOR_TO_FACE.keys())}.")
        parts.extend(mapped)

    cube_str = "".join(parts)

    # Extra validation: each face letter must appear exactly 9 times
    counts = Counter(cube_str)
    for f in "URFDLB":
        if counts[f] != 9:
            raise ValueError(f"Invalid cube: face '{f}' appears {counts[f]} times (expected 9).")

    return cube_str

def normalize_moves(m):
    """
    Ensure we always have a list[str] of moves.
    """
    if m is None:
        return []
    if isinstance(m, str):
        # Handle empty string too
        m = m.strip()
        return m.split() if m else []
    if isinstance(m, (list, tuple)):
        return list(m)
    # Fallback
    return str(m).split()

def human_read_move(move):
    base = move.rstrip("2'")
    suffix = move[len(base):]
    face_map = {"F":"Front", "B":"Back", "U":"Up", "D":"Down", "L":"Left", "R":"Right"}
    face_name = face_map.get(base, base)
    if suffix == "":
        return f"{face_name} — 90° clockwise ({move})"
    if suffix == "'":
        return f"{face_name} — 90° counterclockwise ({move})"
    if suffix == "2":
        return f"{face_name} — 180° ({move})"
    return move

# Init session state
if "moves" not in st.session_state:
    st.session_state.moves = []
if "move_index" not in st.session_state:
    st.session_state.move_index = 0
if "cube_str" not in st.session_state:
    st.session_state.cube_str = None

# Run detection when 6 files are uploaded
if uploaded_files and len(uploaded_files) == 6:
    st.success("6 files uploaded — running color detection...")
    detected_faces = []
    detection_errors = []

    for file in uploaded_files:
        try:
            img_bgr, preview_pil, _ = decode_uploaded_image(file)
            # Your detector likely expects BGR (we pass img_bgr)
            stickers = detect_face_colors_from_image(img_bgr, debug=False)  # returns list of 9 colors 'W','R',...
            if not isinstance(stickers, (list, tuple)) or len(stickers) != 9:
                raise ValueError("Detector did not return 9 stickers.")
            center = stickers[4]  # center sticker
            detected_faces.append({"center": center, "stickers": stickers, "preview": preview_pil})
        except Exception as e:
            detection_errors.append(f"{file.name}: {e}")

    if detection_errors:
        st.error("One or more faces failed to process:")
        for msg in detection_errors:
            st.write("• " + msg)

    if len(detected_faces) == 6:
        # Show detected centers and previews
        cols = st.columns(6)
        for i, d in enumerate(detected_faces):
            with cols[i]:
                st.image(d['preview'], caption=f"Center: {d['center']}")
                st.code(" ".join(d['stickers']))

        # Try to build the cube string
        try:
            cube_str = build_cube_string(detected_faces)
            st.session_state.cube_str = cube_str
            st.subheader("🧩 Cube string (URFDLB order):")
            st.code(cube_str)
        except Exception as e:
            st.session_state.cube_str = None
            st.error(f"Could not construct cube state automatically: {e}")
            st.info("If detection fails, either re-upload clearer photos or enter a manual cube-state below.")

        # Solve if cube_str ready
        if st.session_state.cube_str:
            try:
                raw_moves = solve_kociemba(st.session_state.cube_str)
                st.session_state.moves = normalize_moves(raw_moves)
                st.session_state.move_index = 0

                if not st.session_state.moves:
                    st.warning("Solver returned no moves (cube may already be solved).")

                st.subheader("📌 Solution moves:")
                st.code(" ".join(st.session_state.moves))

                if st.session_state.moves:
                    col1, col2, col3 = st.columns([1,2,1])
                    with col1:
                        if st.button("Prev", key="prev_btn") and st.session_state.move_index > 0:
                            st.session_state.move_index -= 1
                    with col3:
                        if st.button("Next", key="next_btn") and st.session_state.move_index < len(st.session_state.moves) - 1:
                            st.session_state.move_index += 1
                    with col2:
                        st.write(f"Step {st.session_state.move_index + 1} / {len(st.session_state.moves)}")
                        st.markdown(f"### {human_read_move(st.session_state.moves[st.session_state.move_index])}")

                    st.markdown("**Full sequence:**")
                    st.code(" ".join(st.session_state.moves))

            except Exception as e:
                st.error(f"Solver failed: {e}")

# Manual cube state input
st.markdown("---")
st.subheader("Manual input / fallback")
st.markdown("If automatic detection fails, you can paste the 54-character cube string (URFDLB order). Example format: `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`")
manual = st.text_area("Paste 54-character cube string (optional)", height=80)

if manual:
    manual_clean = manual.strip().replace(" ", "").upper()
    if len(manual_clean) != 54 or any(c not in "URFDLB" for c in manual_clean):
        st.error("Invalid manual string. Must be 54 characters consisting only of U,R,F,D,L,B.")
    else:
        # Extra validation: each face letter 9 times
        counts = Counter(manual_clean)
        bad = [f for f in "URFDLB" if counts[f] != 9]
        if bad:
            st.error("Invalid manual string: each of U,R,F,D,L,B must appear exactly 9 times.")
        else:
            st.success("Manual cube string accepted. Solving...")
            try:
                raw_moves = solve_kociemba(manual_clean)
                st.session_state.moves = normalize_moves(raw_moves)
                st.session_state.move_index = 0
                st.session_state.cube_str = manual_clean

                if not st.session_state.moves:
                    st.warning("Solver returned no moves (cube may already be solved).")
                else:
                    st.write("Moves:")
                    st.code(" ".join(st.session_state.moves))
            except Exception as e:
                st.error("Solver error: " + str(e))

st.markdown("<div class='small'>Tip: For best detection, take photos in good lighting, keep the face centered and parallel to camera, and avoid heavy shadows or reflections.</div>", unsafe_allow_html=True)
