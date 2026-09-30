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

# Custom CSS: Clean, full-width, balanced two-column layout
st.markdown(
    """
    <style>
    /* Global Container */
    .block-container {
        max-width: 1140px;
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
    }
    
    /* Typography */
    .app-title {
        font-size: 2.15rem;
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
        margin-bottom: 1.8rem;
        line-height: 1.6;
    }
    .section-header {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #334155;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.35rem;
    }

    /* Seamless Segmented Control (Tabs) */
    div[data-testid="stRadio"] {
        width: 100% !important;
        margin-bottom: 1rem !important;
    }
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        width: 100% !important;
        background-color: #f1f5f9 !important;
        border-radius: 8px !important;
        padding: 3px !important;
        border: 1px solid #e2e8f0 !important;
        gap: 4px !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stRadio"] label {
        flex: 1 1 50% !important;
        margin: 0 !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
        text-align: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        box-sizing: border-box !important;
    }
    /* Active State (Selected Segment) */
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }

    /* Prediction Card */
    .prediction-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.25rem;
        width: 100%;
        box-sizing: border-box;
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

    /* Full-Height Placeholder Box */
    .placeholder-box {
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        min-height: 420px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #64748b;
        background-color: #f8fafc;
        font-size: 0.95rem;
        width: 100%;
        box-sizing: border-box;
        text-align: center;
        padding: 2.5rem 1.5rem;
    }
    
    /* Preprocessing Info Box */
    .prep-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.85rem;
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.5;
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
    'Data Science Capstone Project | รองรับทั้งไฟล์ภาพและลายมือเขียนดิจิทัล'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# State Handling & Instant Mode Switch Management
# ============================================================

MODE_UPLOAD = "อัปโหลดไฟล์ภาพ"
MODE_DRAW = "วาดเขียนลายมือ"

if "mode_counter" not in st.session_state:
    st.session_state.mode_counter = 0
if "sample_image" not in st.session_state:
    st.session_state.sample_image = None


def on_mode_switch():
    """
    Executed synchronously when user clicks between Upload and Draw.
    Increments mode_counter to give input widgets a clean slate instantly,
    guaranteeing no stale images or lingering predictions.
    """
    st.session_state.mode_counter += 1
    st.session_state.sample_image = None


# Load Examples
BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"
example_files = []
if TEST_IMAGES_DIR.exists():
    example_files = sorted(
        [p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:8]


# ============================================================
# 2-Column Interface (Balanced Width: Left Input / Right Output)
# ============================================================

col_input, col_output = st.columns([1, 1], gap="large")

# ----------------- LEFT: INPUT PANEL -----------------
with col_input:
    st.markdown('<div class="section-header">วิธีการนำเข้าข้อมูล (Input Method)</div>', unsafe_allow_html=True)

    # Segmented Radio Bar with callback for instantaneous reset
    selected_mode = st.radio(
        "โหมดการทำงาน",
        options=[MODE_UPLOAD, MODE_DRAW],
        key="selected_mode_key",
        on_change=on_mode_switch,
        horizontal=True,
        label_visibility="collapsed",
    )

    active_image = None
    current_key_suffix = f"_{st.session_state.mode_counter}"

    # MODE 1: FILE UPLOAD (PRIMARY / DEFAULT)
    if selected_mode == MODE_UPLOAD:
        st.caption("เลือกไฟล์ภาพพยัญชนะไทยจากเครื่องคอมพิวเตอร์:")
        uploaded_file = st.file_uploader(
            "Upload Image File",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"upload_widget{current_key_suffix}",
        )
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file)
            st.image(active_image, caption=f"ไฟล์ภาพ: {uploaded_file.name}", use_container_width=True)

    # MODE 2: DRAWING CANVAS (SECONDARY)
    else:
        st.caption("วาดพยัญชนะไทยลงในกรอบสี่เหลี่ยมด้านล่าง:")

        col_slider, col_hint = st.columns([2, 1])
        with col_slider:
            stroke_width = st.slider("ขนาดเส้น:", min_value=8, max_value=24, value=14, step=2)
        with col_hint:
            st.write("")
            st.caption("(ดับเบิลคลิกถังขยะเพื่อล้าง)")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_widget{current_key_suffix}",
        )

        if canvas_result.image_data is not None:
            rgb = canvas_result.image_data[:, :, :3]
            if np.any(rgb < 200):
                active_image = Image.fromarray(rgb.astype(np.uint8))
                st.caption("สถานะ: ตรวจพบภาพวาด กำลังประมวลผลการทำนาย")

    # If user selected an example image from the gallery
    if st.session_state.sample_image is not None and active_image is None:
        active_image = st.session_state.sample_image
        st.image(active_image, caption="ภาพจากชุดตัวอย่างทดสอบ", use_container_width=True)
        if st.button("ยกเลิกภาพตัวอย่าง", use_container_width=True):
            st.session_state.sample_image = None
            st.rerun()


# ----------------- RIGHT: OUTPUT PANEL (FULL-HEIGHT & FULL-WIDTH) -----------------
with col_output:
    st.markdown('<div class="section-header">ผลการวิเคราะห์และจำแนก (Classification Results)</div>', unsafe_allow_html=True)

    if active_image is not None:
        try:
            scores, preview = predictor.predict(active_image)
            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

            # Formal academic slate/blue tones
            bar_colors = ["#1e40af", "#3b82f6", "#64748b", "#94a3b8", "#cbd5e1"]

            # Prediction Card (Full Width)
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

            # Processed 28x28 Image & Pipeline details
            st.markdown('<div class="section-header" style="margin-top: 1.4rem;">ภาพหลังกระบวนการเตรียมข้อมูล (Processed 28 × 28 Image)</div>', unsafe_allow_html=True)
            
            col_prev_img, col_prev_info = st.columns([1, 2], gap="medium")
            with col_prev_img:
                st.image(
                    preview,
                    caption="ขนาด 28 × 28 พิกเซล",
                    use_container_width=True,
                )
            with col_prev_info:
                st.markdown(
                    """
                    <div class="prep-box">
                        <b>กระบวนการเตรียมข้อมูล (Image Pipeline):</b><br>
                        1. แปลงระดับสีเทา (Grayscale 0-255)<br>
                        2. ตรวจจับขอบเขตและตัดขอบ (Auto-crop)<br>
                        3. จัดตำแหน่งกึ่งกลางภาพ (Centering + 4px Padding)<br>
                        4. ปรับขนาดมาตรฐานเป็น 28 × 28 พิกเซล
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
    else:
        # Full-height placeholder matching left side
        st.markdown(
            """
            <div class="placeholder-box">
                <div style="font-size: 1.1rem; font-weight: 600; color: #334155; margin-bottom: 0.5rem;">
                    รอข้อมูลนำเข้าเพื่อวิเคราะห์ผล
                </div>
                <div style="color: #64748b; font-size: 0.9rem; max-width: 320px; line-height: 1.6;">
                    กรุณาเลือกอัปโหลดไฟล์ภาพ หรือสลับไปยังโหมดวาดเขียนลายมือทางฝั่งซ้าย เพื่อดูผลการจำแนกพยัญชนะ
                </div>
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
                st.session_state.sample_image = img_ex
                st.rerun()

# Footer
st.write("---")
st.caption(
    "Data Science Capstone Project | Model: Deep Convolutional Neural Network (Accuracy: 99.14%)"
)
