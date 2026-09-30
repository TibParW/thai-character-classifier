from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st

from cnn_model import ThaiCNNPredictor
from preprocessing import preprocess_image

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Thai Character Classifier (ก - ฮ)",
    page_icon="🇹🇭",
    layout="centered",
)

st.title("🇹🇭 Thai Character Classifier (ก - ฮ)")
st.write(
    "### Data Science Capstone Project: Image Classification (2.2)\n"
    "ระบบปัญญาประดิษฐ์สำหรับจำแนกพยัญชนะไทย 44 ตัว ทั้ง**ตัวพิมพ์ (Printed Fonts)** "
    "และ**ลายมือเขียน (Handwritten)** ด้วยโมเดล Deep CNN (ความแม่นยำ 99.14%)"
)


# ============================================================
# Load Model (Cached)
# ============================================================
@st.cache_resource
def load_predictor():
    return ThaiCNNPredictor()


predictor = load_predictor()

# ============================================================
# Input: File Upload or Camera
# ============================================================
tab1, tab2 = st.tabs(["📁 อัปโหลดรูปภาพ", "📷 ถ่ายภาพ"])

uploaded_file = None
with tab1:
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพพยัญชนะไทย (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
    )

with tab2:
    camera_file = st.camera_input("ถ่ายภาพพยัญชนะไทยผ่านกล้อง")
    if camera_file is not None:
        uploaded_file = camera_file

# Sample buttons
st.write("---")
st.write("💡 หรือเลือกทดสอบจากตัวอย่างภาพ:")
sample_dir = Path(__file__).resolve().parent / "test_images"
cols = st.columns(4)

selected_sample = None
if sample_dir.exists():
    sample_files = sorted(
        [p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:8]
    for idx, sample_p in enumerate(sample_files):
        with cols[idx % 4]:
            if st.button(f"ตัวอย่าง {idx+1}", key=f"sample_{idx}"):
                selected_sample = sample_p

# Determine which image to process
img_to_process = None
if uploaded_file is not None:
    img_to_process = Image.open(uploaded_file)
elif selected_sample is not None:
    img_to_process = Image.open(selected_sample)

# ============================================================
# Prediction & Display
# ============================================================
if img_to_process is not None:
    st.write("---")
    col_img, col_pred = st.columns(2)

    with col_img:
        st.subheader("ภาพนำเข้า")
        st.image(img_to_process, width=220)

    # Predict
    with st.spinner("กำลังประมวลผลและทำนาย..."):
        scores, preview = predictor.predict(img_to_process)

    # Sort top 5
    top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_char, top_prob = top_5[0]

    with col_pred:
        st.subheader("ผลการทำนายอันดับ 1")
        st.success(f"### **{top_char}**\n**ความมั่นใจ: {top_prob * 100:.2f}%**")

        st.caption("ภาพที่ผ่าน Preprocessing (28 × 28 พิกเซล):")
        st.image(preview, width=100)

    st.write("#### 📊 ความน่าจะเป็น 5 อันดับแรก (Top-5 Probabilities)")
    df_top5 = pd.DataFrame(
        [{"พยัญชนะ": k, "ความมั่นใจ (%)": v * 100} for k, v in top_5]
    )
    st.bar_chart(df_top5.set_index("พยัญชนะ"))
    st.dataframe(df_top5, use_container_width=True)

# Footer
st.write("---")
st.caption(
    "โครงงาน Data Science Capstone Project | โมเดล: Deep Convolutional Neural Network (CNN) 22,000 ตัวอย่าง"
)
