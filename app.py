from pathlib import Path
import gradio as gr
import joblib
import torch

from inference import ThaiCharacterPredictor
from cnn_model import ThaiCNNPredictor

# ============================================================
# Paths & Models
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = BASE_DIR / "models" / "thai_cnn_classifier.pt"
RF_MODEL_PATH = BASE_DIR / "models" / "thai_character_classifier.joblib"
TEST_IMAGES_DIR = BASE_DIR / "test_images"

# Load Deep CNN Predictor (99.41% Accuracy)
cnn_predictor = None
if CNN_MODEL_PATH.exists():
    print(f"Loading Deep CNN model from: {CNN_MODEL_PATH}")
    cnn_predictor = ThaiCNNPredictor(CNN_MODEL_PATH)
else:
    print(f"[Warning] Deep CNN model not found at {CNN_MODEL_PATH}")

# Load Random Forest Predictor (Baseline ML)
rf_predictor = None
if RF_MODEL_PATH.exists():
    print(f"Loading Random Forest model from: {RF_MODEL_PATH}")
    rf_predictor = ThaiCharacterPredictor(joblib.load(RF_MODEL_PATH))


def predict_router(image, model_choice):
    """
    Route prediction to either Deep CNN (99.4% Accuracy) or Random Forest (Baseline ML).
    """
    if image is None:
        raise gr.Error("กรุณาอัปโหลดรูปภาพหรือวาดตัวอักษรไทยก่อนกดทำนาย")

    if "CNN" in model_choice and cnn_predictor is not None:
        return cnn_predictor.predict(image)
    elif rf_predictor is not None:
        return rf_predictor.predict(image)
    elif cnn_predictor is not None:
        return cnn_predictor.predict(image)
    else:
        raise FileNotFoundError("ไม่พบไฟล์โมเดล กรุณารัน train_cnn.py หรือ train_model.py ก่อน")


# Collect sample test images
example_images = []
if TEST_IMAGES_DIR.exists():
    example_images = sorted(
        [[str(p), "🧠 Deep CNN Model (แนะนำ - แม่นยำ 99.41% แยกได้ทั้งตัวพิมพ์และลายมือ)"] 
         for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

# ============================================================
# Gradio Interface
# ============================================================

title = "🇹🇭 ระบบจำแนกพยัญชนะไทย 44 ตัว (Thai Character Recognition AI)"
description = """
### Data Science Capstone Project: Image Classification Model (2.2) & Web App (3)
ระบบปัญญาประดิษฐ์สำหรับจำแนกพยัญชนะภาษาไทยทั้ง **44 ตัว (ก - ฮ)** 
รองรับทั้ง **ภาพตัวพิมพ์คอมพิวเตอร์ (Printed Fonts 14 แบบ)** และ **ตัวหนังสือลายมือเขียนจริง (Handwritten Characters)**
ผ่านกระบวนการ Preprocessing จัดกึ่งกลางภาพ (Auto-crop & Centering) และฝึกสอนด้วยชุดข้อมูลกว่า **11,000 ภาพ**

**วิธีใช้งาน:**
1. **อัปโหลดภาพ หรือ วาดตัวอักษร** ลงในช่องด้านซ้าย
2. **เลือกแบบจำลอง:** แนะนำให้ใช้ **Deep CNN (99.41%)** เพื่อความฉลาดสูงสุด หรือสลับเป็น **Random Forest** เพื่อเปรียบเทียบ
3. ระบบจะแสดง **5 อันดับพยัญชนะที่มีความน่าจะเป็นสูงสุด** พร้อมภาพ 28×28 พิกเซลที่ผ่านการเตรียมข้อมูล
"""

article = """
---
### สรุปผลการประเมินประสิทธิภาพ (Model Performance Comparison):
- **🧠 Deep Convolutional Neural Network (CNN):** **Test Accuracy: 99.41%** (Macro F1: 0.9940)  
  *สามารถแยกแยะคู่ตัวอักษรที่มีลายเส้นคล้ายกัน เช่น ข/ช, บ/ป, ฎ/ฏ, ญ/ณ ได้อย่างขาดลอย*
- **⚡ Random Forest Classifier (Baseline ML):** **Test Accuracy: 92.64%** (Macro F1: 0.9258)

**จัดทำขึ้นเพื่อ:** โครงการวิชาวิทยาศาสตร์ข้อมูล (Data Science Capstone Project)
"""

with gr.Blocks(title=title, theme=gr.themes.Soft()) as app:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                sources=["upload", "clipboard"],
                type="numpy",
                label="ภาพนำเข้า (อัปโหลด หรือ วางภาพพยัญชนะไทย)",
            )
            model_selector = gr.Radio(
                choices=[
                    "🧠 Deep CNN Model (แนะนำ - แม่นยำ 99.41% แยกได้ทั้งตัวพิมพ์และลายมือ)",
                    "⚡ Random Forest Model (Baseline ML - 92.64%)",
                ],
                value="🧠 Deep CNN Model (แนะนำ - แม่นยำ 99.41% แยกได้ทั้งตัวพิมพ์และลายมือ)",
                label="เลือกแบบจำลองที่ต้องการใช้งาน (Model Selection)",
            )
            predict_btn = gr.Button("🔍 จำแนกตัวอักษร (Predict)", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_label = gr.Label(
                num_top_classes=5,
                label="ผลการทำนาย 5 อันดับแรก (Top-5 Predictions)",
            )
            output_preview = gr.Image(
                type="numpy",
                label="ภาพที่โมเดลมองเห็นหลังทำ Preprocessing (28 × 28 Pixels)",
            )

    predict_btn.click(
        fn=predict_router,
        inputs=[input_image, model_selector],
        outputs=[output_label, output_preview],
    )

    if example_images:
        gr.Examples(
            examples=example_images[:10],
            inputs=[input_image, model_selector],
            label="ตัวอย่างภาพสำหรับทดลองคลิกตรวจงาน (Click to Test Samples)",
        )

    gr.Markdown(article)

if __name__ == "__main__":
    # ตั้งค่า share=True เพื่อรับลิงก์สาธารณะใช้งานผ่านเน็ต 72 ชม.
    app.launch(share=False)
