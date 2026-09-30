from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from cnn_model import ThaiCNNPredictor
from preprocessing import preprocess_image


# ============================================================
# Page Configuration & Professional Academic Styling
# ============================================================

st.set_page_config(
    page_title="Thai Character Classifier (ก - ฮ)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS: Formal, minimal, professional, no emojis, elegant segmented tab
st.markdown(
    """
    <style>
    /* Global Container */
    .block-container {
        max-width: 1040px;
        padding-top: 2rem;
        padding-bottom: 3.5rem;
    }
    
    /* Typography */
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.35rem;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        text-align: center;
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .section-header {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #334155;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.4rem;
    }

    /* Segmented Control Styling */
    div[data-testid="stSegmentedControl"] {
        width: 100%;
        margin-bottom: 1.25rem;
    }
    div[data-testid="stSegmentedControl"] button {
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    /* Custom Segmented Buttons for Radio fallback */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: row;
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
    }
    div[data-testid="stRadio"] label {
        flex: 1;
        text-align: center;
        border-radius: 6px;
        padding: 8px 14px;
        margin: 0;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 500;
        color: #475569;
        transition: all 0.15s ease-in-out;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background-color: #ffffff;
        color: #0f172a;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    /* Prediction Card */
    .prediction-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.25rem;
    }
    .prob-row {
        margin-bottom: 0.65rem;
    }
    .prob-row:last-child {
        margin-bottom: 0;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        color: #1e293b;
    }
    .prob-bar-bg {
        background-color: #f1f5f9;
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    /* Canvas Frame */
    .canvas-wrapper {
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        display: inline-block;
        background-color: #ffffff;
    }
    
    /* Placeholder Box */
    .placeholder-box {
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        padding: 4.5rem 1.5rem;
        text-align: center;
        color: #64748b;
        background-color: #f8fafc;
        font-size: 0.92rem;
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
# Header (Formal & Academic)
# ============================================================

st.markdown('<div class="app-title">Thai Character Classification System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'ระบบจำแนกพยัญชนะภาษาไทย 44 รูป (ก - ฮ) ด้วยโครงข่ายประสาทเทียมสังวัตนาการ (Deep Convolutional Neural Network)<br>'
    'Data Science Capstone Project | ทดสอบได้ทั้งลายมือเขียนสดบนกระดานดิจิทัลและไฟล์ภาพ'
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
# 2-Column Interface
# ============================================================

col_input, col_output = st.columns([1, 1], gap="large")

active_image = None

# ----------------- LEFT: INPUT PANEL -----------------
with col_input:
    st.markdown('<div class="section-header">วิธีการนำเข้าข้อมูล (Input Method)</div>', unsafe_allow_html=True)

    # Segmented Control / Tab (Left & Right)
    options = ["วาดเขียนด้วยลายมือ (Drawing Canvas)", "อัปโหลดไฟล์รูปภาพ (Upload Image)"]
    if hasattr(st, "segmented_control"):
        selected_mode = st.segmented_control(
            "Input Mode",
            options=options,
            default=options[0],
            label_visibility="collapsed",
        )
        if not selected_mode:
            selected_mode = options[0]
    else:
        selected_mode = st.radio(
            "Input Mode",
            options=options,
            horizontal=True,
            label_visibility="collapsed",
        )

    # MODE 1: DRAWING CANVAS
    if selected_mode == options[0]:
        st.caption("วาดพยัญชนะไทยลงในกรอบสี่เหลี่ยมด้านล่าง:")

        col_slider, col_hint = st.columns([2, 1])
        with col_slider:
            stroke_width = st.slider("ขนาดเส้น (Stroke):", min_value=8, max_value=24, value=14, step=2)
        with col_hint:
            st.write("")
            st.caption("(ดับเบิลคลิกไอคอนถังขยะเพื่อล้างภาพ)")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="formal_drawing_canvas",
        )

        if canvas_result.image_data is not None:
            rgb = canvas_result.image_data[:, :, :3]
            if np.any(rgb < 200):
                active_image = Image.fromarray(rgb.astype(np.uint8))
                st.caption("สถานะ: ตรวจพบภาพวาด กำลังประมวลผลการทำนาย")

    # MODE 2: FILE UPLOAD
    else:
        st.caption("เลือกไฟล์ภาพพยัญชนะไทยจากเครื่องคอมพิวเตอร์:")
        uploaded_file = st.file_uploader(
            "Upload Image File",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="file_uploader_input",
        )
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file)
            st.image(active_image, caption=f"ไฟล์ภาพ: {uploaded_file.name}", use_container_width=True)

    # Fallback to sample click
    if st.session_state.current_image is not None and active_image is None:
        active_image = st.session_state.current_image
        st.image(active_image, caption="ภาพจากชุดตัวอย่างทดสอบ", use_container_width=True)


# ----------------- RIGHT: OUTPUT PANEL -----------------
with col_output:
    st.markdown('<div class="section-header">ผลการวิเคราะห์และจำแนก (Classification Results)</div>', unsafe_allow_html=True)

    if active_image is not None:
        try:
            scores, preview = predictor.predict(active_image)
            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

            # Professional slate / blue gradient palette (formal, not rainbow)
            bar_colors = ["#1e40af", "#3b82f6", "#64748b", "#94a3b8", "#cbd5e1"]

            # Prediction Card
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            for idx, (char_label, prob) in enumerate(top_5):
                pct = prob * 100.0
                color = bar_colors[idx]
                font_weight = "700" if idx == 0 else "500"
                font_size = "1.02rem" if idx == 0 else "0.88rem"

                row_html = f"""
                <div class="prob-row">
                    <div class="prob-header" style="font-weight: {font_weight}; font-size: {font_size};">
                        <span>{idx+1}. {char_label}</span>
                        <span>{pct:.2f}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {max(pct, 1.2):.2f}%; background-color: {color};"></div>
                    </div>
                </div>
                """
                st.markdown(row_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Processed 28x28 Image
            st.markdown('<div class="section-header" style="margin-top: 1.4rem;">ภาพหลังกระบวนการเตรียมข้อมูล (Processed 28 × 28 Image)</div>', unsafe_allow_html=True)
            st.image(
                preview,
                caption="ภาพมาตรฐานขนาด 28 × 28 พิกเซล (Grayscale, Centered & Cropped)",
                width=150,
            )

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
    else:
        st.markdown(
            """
            <div class="placeholder-box">
                <b>รอข้อมูลนำเข้า</b><br>
                กรุณาวาดตัวอักษรลงบนกระดาน หรืออัปโหลดไฟล์ภาพทางฝั่งซ้าย
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Examples (ชุดภาพตัวอย่างสำหรับทดสอบ)
# ============================================================

if example_files:
    st.write("---")
    st.markdown('<div class="section-header">ชุดภาพตัวอย่างสำหรับทดสอบ (Sample Test Images)</div>', unsafe_allow_html=True)

    num_cols = min(len(example_files), 8)
    cols = st.columns(num_cols)

    for idx, ex_path in enumerate(example_files):
        with cols[idx]:
            img_ex = Image.open(ex_path)
            st.image(img_ex, use_container_width=True)
            if st.button(f"ตัวอย่าง {idx+1}", key=f"btn_ex_{idx}", use_container_width=True):
                st.session_state.current_image = img_ex
                st.rerun()

# Footer
st.write("---")
st.caption(
    "Data Science Capstone Project | Model: Deep Convolutional Neural Network (Accuracy: 99.14%)"
)
