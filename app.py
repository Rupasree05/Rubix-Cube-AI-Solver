# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

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

def read_uploaded_img(uploaded_file):
    data = uploaded_file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def build_cube_string(detected_faces):
    """
    detected_faces: list of dicts [{'center': 'W', 'stickers': ['W','R',...'], 'img': PIL or array}, ...]
    Returns 54-char string or raises error.
    """
    # Map center color -> stickers
    center_map = {d['center']: d['stickers'] for d in detected_faces}
    # Validate we have all 6 centers
    needed_centers = set(FACE_CENTER.values())  # {'W','R','G','Y','O','B'}
    if set(center_map.keys()) != needed_centers:
        raise ValueError(f"Detected centers mismatch. Found: {set(center_map.keys())}. Needed: {needed_centers}")
    parts = []
    for face in face_order:
        center_color = FACE_CENTER[face]  # e.g. 'W' for U
        stickers = center_map[center_color]  # list of 9 color letters
        # Convert color letters (W,Y,R,...) to face letters (U,R,F,...) expected by kociemba
        mapped = [COLOR_TO_FACE[c] for c in stickers]
        parts.extend(mapped)
    return "".join(parts)

# Run detection when 6 files are uploaded
if uploaded_files and len(uploaded_files) == 6:
    st.success("6 files uploaded — running color detection...")
    detected_faces = []
    for file in uploaded_files:
        img = read_uploaded_img(file)  # BGR
        stickers = detect_face_colors_from_image(img, debug=False)  # returns list of 9 colors 'W','R',...
        center = stickers[4]  # center sticker
        # Save a small preview
        pil = Image.open(BytesIO(file.read())) if False else Image.open(file)  # file may be exhausted; using file directly works in streamlit
        # But streamlit's uploaded_file supports .getvalue or we could re-read; safer: rebuild from cv2->RGB
        # Build a small preview from cv2 image
        preview = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preview_pil = Image.fromarray(preview)
        detected_faces.append({"center": center, "stickers": stickers, "preview": preview_pil})

    # Show detected centers and previews
    cols = st.columns(6)
    for i, d in enumerate(detected_faces):
        with cols[i]:
            st.image(d['preview'], caption=f"Center: {d['center']}")
            st.write(" ".join(d['stickers']))

    # Try to build the cube string
    try:
        cube_str = build_cube_string(detected_faces)
        st.subheader("🧩 Cube string (URFDLB order):")
        st.code(cube_str)
    except Exception as e:
        st.error(f"Could not construct cube state automatically: {e}")
        st.info("If detection fails, either re-upload clearer photos or enter a manual cube-state below.")
        cube_str = None

    if cube_str:
        # Solve
        try:
            moves = solve_kociemba(cube_str)
            st.subheader("📌 Solution moves:")
            st.write(" ".join(moves))

            # Step-by-step navigation using session state
            if 'move_index' not in st.session_state:
                st.session_state.move_index = 0

            def human_read_move(move):
                base = move.rstrip("2'")
                suffix = move[len(base):]
                face_map = {
                    "F":"Front", "B":"Back", "U":"Up", "D":"Down", "L":"Left", "R":"Right"
                }
                face_name = face_map.get(base, base)
                if suffix == "":
                    return f"{face_name} — 90° clockwise ({move})"
                if suffix == "'":
                    return f"{face_name} — 90° counterclockwise ({move})"
                if suffix == "2":
                    return f"{face_name} — 180° ({move})"
                return move

            # controls
            col1, col2, col3 = st.columns([1,2,1])
            with col1:
                if st.button("Prev") and st.session_state.move_index > 0:
                    st.session_state.move_index -= 1
            with col3:
                if st.button("Next") and st.session_state.move_index < len(moves) - 1:
                    st.session_state.move_index += 1
            with col2:
                st.write(f"Step {st.session_state.move_index + 1} / {len(moves)}")
                st.markdown(f"### {human_read_move(moves[st.session_state.move_index])}")

            # show full move list
            st.markdown("**Full sequence:**")
            st.code(" ".join(moves))

        except Exception as e:
            st.error(f"Solver failed: {e}")

# Manual cube state input
st.markdown("---")
st.subheader("Manual input / fallback")
st.markdown("If automatic detection fails, you can paste the 54-character cube string (URFDLB order). Example format: `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`")
manual = st.text_area("Paste 54-character cube string (optional)", height=80)
if manual:
    manual = manual.strip().replace(" ", "").upper()
    if len(manual) != 54 or any(c not in "URFDLB" for c in manual):
        st.error("Invalid manual string. Must be 54 characters consisting of U,R,F,D,L,B.")
    else:
        st.success("Manual cube string accepted. Solving...")
        try:
            moves = solve_kociemba(manual)
            st.write("Moves:", " ".join(moves))
            # reset move index
            st.session_state.move_index = 0
        except Exception as e:
            st.error("Solver error: " + str(e))

st.markdown("<div class='small'>Tip: For best detection, take photos in good lighting, keep the face centered and parallel to camera, and avoid heavy shadows or reflections.</div>", unsafe_allow_html=True)
