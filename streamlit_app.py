import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import streamlit as st
import numpy as np

from cnn_model import ThaiCNNPredictor
from preprocessing import preprocess_image


# ============================================================
# Page Configuration & Gradio-Like Styling
# ============================================================

st.set_page_config(
    page_title="Thai Character Classifier (ก - ฮ)",
    page_icon="🇹🇭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS to replicate Gradio gr.Interface look & feel
st.markdown(
    """
    <style>
    /* Main container max width */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header typography */
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
        color: #1f2937;
    }
    .app-desc {
        text-align: center;
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    
    /* Card panel styling similar to Gradio */
    .gradio-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #4b5563;
        margin-bottom: 0.75rem;
    }
    
    /* Gradio Label / Progress Bar Styling */
    .gradio-label-row {
        margin-bottom: 0.6rem;
    }
    .label-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: #1f2937;
    }
    .bar-bg {
        background-color: #f3f4f6;
        border-radius: 9999px;
        height: 12px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease-in-out;
    }
    
    /* Example thumbnail gallery */
    .example-btn-container {
        text-align: center;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load Model (Cached)
# ============================================================

@st.cache_resource
def load_predictor():
    return ThaiCNNPredictor()

predictor = load_predictor()


# ============================================================
# Title & Description (Gradio gr.Interface style)
# ============================================================

st.markdown('<div class="app-title">Thai Character Classifier (ก - ฮ)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-desc">'
    'Upload an image containing one Thai consonant from ก to ฮ. '
    'The application preprocesses the image and uses a trained Deep CNN model (99.14% accuracy on 22,000 samples) to classify the character.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# State Handling for Examples & Inputs
# ============================================================

if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = ""


# Load Example Images
BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"
example_files = []
if TEST_IMAGES_DIR.exists():
    example_files = sorted(
        [p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:8]


# ============================================================
# 2-Column Interface (1:1 identical to Gradio gr.Interface)
# ============================================================

col_input, col_output = st.columns([1, 1], gap="large")

# ----------------- LEFT: INPUT PANEL -----------------
with col_input:
    st.markdown('<div class="card-title">Upload a Thai character (อัปโหลดรูปพยัญชนะไทย)</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file)
        st.session_state.current_image_name = uploaded_file.name

    # Display Input Image
    if st.session_state.current_image is not None:
        st.image(
            st.session_state.current_image,
            caption=f"Input Image: {st.session_state.current_image_name}",
            use_container_width=True,
        )
        if st.button("🗑️ Clear Image", use_container_width=True):
            st.session_state.current_image = None
            st.session_state.current_image_name = ""
            st.rerun()
    else:
        st.info("👆 ลากรูปภาพมาวาง หรือคลิกเพื่ออัปโหลด หรือกดเลือกจาก Examples ด้านล่าง")


# ----------------- RIGHT: OUTPUT PANEL -----------------
with col_output:
    st.markdown('<div class="card-title">Prediction (ผลการทำนาย 5 อันดับแรก)</div>', unsafe_allow_html=True)
    
    if st.session_state.current_image is not None:
        try:
            scores, preview = predictor.predict(st.session_state.current_image)
            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Palette for Gradio-like bars
            bar_colors = ["#ff7c00", "#3b82f6", "#10b981", "#8b5cf6", "#6b7280"]

            # Render Gradio-style Label bars
            st.markdown('<div class="gradio-card">', unsafe_allow_html=True)
            for idx, (char_label, prob) in enumerate(top_5):
                pct = prob * 100.0
                color = bar_colors[idx] if idx < len(bar_colors) else "#9ca3af"
                weight = "700" if idx == 0 else "500"
                font_size = "1.05rem" if idx == 0 else "0.9rem"
                
                bar_html = f"""
                <div class="gradio-label-row">
                    <div class="label-header" style="font-weight: {weight}; font-size: {font_size};">
                        <span>{char_label}</span>
                        <span>{pct:.2f}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width: {max(pct, 1.5):.2f}%; background-color: {color};"></div>
                    </div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Processed 28x28 Image
            st.markdown('<div class="card-title" style="margin-top: 1.5rem;">Processed 28 × 28 image (ภาพหลังทำ Preprocessing)</div>', unsafe_allow_html=True)
            st.image(
                preview,
                caption="Grayscale, Centered & Resized (28 × 28 pixels)",
                width=160,
            )

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
    else:
        # Placeholder when waiting for input (like Gradio)
        st.markdown(
            """
            <div style="border: 2px dashed #e5e7eb; border-radius: 8px; padding: 3rem 1rem; text-align: center; color: #9ca3af;">
                รอการอัปโหลดรูปภาพทางด้านซ้าย...<br><span style="font-size: 0.85rem;">(Waiting for image upload)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Examples (Gradio gr.Examples style)
# ============================================================

if example_files:
    st.write("---")
    st.markdown('<div class="card-title">Examples (ตัวอย่างภาพสำหรับคลิกทดสอบทันที)</div>', unsafe_allow_html=True)
    
    num_cols = min(len(example_files), 8)
    cols = st.columns(num_cols)
    
    for idx, ex_path in enumerate(example_files):
        with cols[idx]:
            img_ex = Image.open(ex_path)
            st.image(img_ex, use_container_width=True)
            if st.button(f"ภาพที่ {idx+1}", key=f"btn_ex_{idx}", use_container_width=True):
                st.session_state.current_image = img_ex
                st.session_state.current_image_name = ex_path.name
                st.rerun()

# Footer
st.write("---")
st.caption(
    "Data Science Capstone Project | Model: Deep CNN (22,000 samples, 99.14% accuracy) | Template: Gradio Interface Replica"
)
