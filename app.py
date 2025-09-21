import streamlit as st
from vision import detect_face_colors
from solver import solve_cube

# Page settings
st.set_page_config(page_title="Rubik's Cube AI Solver", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #FFD700;
        margin-bottom: 20px;
    }
    .upload-box {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    }
    .solution {
        background: #1e293b;
        padding: 15px;
        border-radius: 10px;
        font-size: 1.2em;
        color: #00ffcc;
    }
    </style>
""", unsafe_allow_html=True)

# Title with custom HTML
st.markdown('<div class="title">🤖 Rubik\'s Cube AI Solver</div>', unsafe_allow_html=True)

# Upload box
st.markdown('<div class="upload-box">Upload 6 cube face images (JPG/PNG)</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("", accept_multiple_files=True, type=["jpg","png","jpeg"])

if uploaded_files and len(uploaded_files) == 6:
    face_order = ["U","R","F","D","L","B"]
    cube_state = ""

    for i, file in enumerate(uploaded_files):
        with open(f"temp_face_{i}.jpg", "wb") as f:
            f.write(file.read())
        colors = detect_face_colors(f"temp_face_{i}.jpg")
        cube_state += colors
        st.image(f"temp_face_{i}.jpg", caption=f"Face {face_order[i]} - {colors}")

    st.subheader("🧩 Cube State:")
    st.code(cube_state)

    st.markdown('<div class="solution">📌 Solution Steps:</div>', unsafe_allow_html=True)
    solution = solve_cube(cube_state)
    st.write(" → ".join(solution))
