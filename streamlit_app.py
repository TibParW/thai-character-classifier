from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from cnn_model import ThaiCNNPredictor
from preprocessing import preprocess_image


# ============================================================
# Page Configuration & Styling (Gradio Replica)
# ============================================================

st.set_page_config(
    page_title="Thai Character Classifier (ก - ฮ)",
    page_icon="🇹🇭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean interface
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
        color: #111827;
    }
    .app-desc {
        text-align: center;
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }
    .card-title {
        font-size: 0.92rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #1f2937;
        margin-bottom: 0.75rem;
    }
    .gradio-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .gradio-label-row {
        margin-bottom: 0.6rem;
    }
    .label-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.92rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: #1f2937;
    }
    .bar-bg {
        background-color: #f3f4f6;
        border-radius: 9999px;
        height: 13px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease-in-out;
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
# Header
# ============================================================

st.markdown('<div class="app-title">Thai Character Classifier (ก - ฮ)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-desc">'
    'ทดสอบจำแนกพยัญชนะไทย 44 ตัว ทั้งจาก<b>การวาดเขียนด้วยลายมือสด ๆ</b> หรือ<b>การอัปโหลดไฟล์ภาพ</b><br>'
    'ขับเคลื่อนด้วยโมเดล Deep CNN (ความแม่นยำ 99.14% จากชุดข้อมูล 22,000 ตัวอย่าง)'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# State Handling
# ============================================================

if "current_image" not in st.session_state:
    st.session_state.current_image = None


# Load Examples
BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"
example_files = []
if TEST_IMAGES_DIR.exists():
    example_files = sorted(
        [p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:8]


# ============================================================
# 2-Column Interface (Left: Input / Right: Output)
# ============================================================

col_input, col_output = st.columns([1, 1], gap="large")

active_image = None

# ----------------- LEFT: INPUT PANEL -----------------
with col_input:
    st.markdown('<div class="card-title">1. เลือกวิธีป้อนข้อมูล (Input Method)</div>', unsafe_allow_html=True)
    
    # Prominent Radio Button - Default to DRAWING CANVAS
    input_mode = st.radio(
        "โหมดการทดสอบ:",
        ["✏️ วาดเขียนด้วยลายมือ (Drawing Canvas)", "📁 อัปโหลดไฟล์รูปภาพ (Upload Image)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "✏️ วาดเขียนด้วยลายมือ (Drawing Canvas)":
        st.write("🖌️ **กระดานวาดตัวอักษร:** ใช้เมาส์หรือนิ้วมือวาดพยัญชนะไทยลงในกรอบด้านล่าง:")
        
        col_stroke, col_clear = st.columns([2, 1])
        with col_stroke:
            stroke_width = st.slider("ขนาดเส้นดินสอ (Stroke):", min_value=8, max_value=24, value=14, step=2)
        with col_clear:
            st.write("")
            st.caption("*(ดับเบิ้ลคลิกถังขยะในเครื่องมือ เพื่อล้างกระดาน)*")

        # Interactive Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="thai_drawing_canvas",
        )

        # Check if user has drawn something
        if canvas_result.image_data is not None:
            rgb = canvas_result.image_data[:, :, :3]
            # Check if there are non-white pixels
            if np.any(rgb < 200):
                active_image = Image.fromarray(rgb.astype(np.uint8))
                st.success("✅ ได้รับภาพวาดเรียบร้อย กำลังวิเคราะห์ผล...")

    else:
        # File Upload Mode
        st.write("📁 **อัปโหลดภาพ:** เลือกรูปภาพพยัญชนะไทยจากเครื่องหรือ Desktop:")
        uploaded_file = st.file_uploader(
            "เลือกไฟล์รูปภาพ (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="file_uploader",
        )
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file)
            st.image(active_image, caption=f"ภาพที่อัปโหลด: {uploaded_file.name}", use_container_width=True)

    # If an example was clicked from below
    if st.session_state.current_image is not None and active_image is None:
        active_image = st.session_state.current_image
        st.image(active_image, caption="ภาพจาก Examples", use_container_width=True)


# ----------------- RIGHT: OUTPUT PANEL -----------------
with col_output:
    st.markdown('<div class="card-title">2. ผลการทำนาย (Prediction & Preprocessing)</div>', unsafe_allow_html=True)
    
    if active_image is not None:
        try:
            scores, preview = predictor.predict(active_image)
            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            bar_colors = ["#ff7c00", "#3b82f6", "#10b981", "#8b5cf6", "#6b7280"]

            # Render Gradio-style Label bars
            st.markdown('<div class="gradio-card">', unsafe_allow_html=True)
            for idx, (char_label, prob) in enumerate(top_5):
                pct = prob * 100.0
                color = bar_colors[idx] if idx < len(bar_colors) else "#9ca3af"
                weight = "700" if idx == 0 else "500"
                font_size = "1.08rem" if idx == 0 else "0.9rem"
                
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
                caption="Auto-crop, Centered & Resized (28 × 28 pixels)",
                width=160,
            )

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
    else:
        st.markdown(
            """
            <div style="border: 2px dashed #d1d5db; border-radius: 10px; padding: 4.5rem 1rem; text-align: center; color: #9ca3af; background: #fafafa;">
                ✍️ <b>รอยวาดหรืออัปโหลดรูปภาพทางด้านซ้าย...</b><br>
                <span style="font-size: 0.85rem;">(Draw a character on the canvas or upload an image to see prediction)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Examples (ตัวอย่างภาพ)
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
                st.rerun()

# Footer
st.write("---")
st.caption(
    "Data Science Capstone Project | Model: Deep CNN (22,000 samples, 99.14% accuracy) | Dual Mode: Live Canvas & Upload"
)
