from pathlib import Path

import gradio as gr
import numpy as np
import torch

from cnn_model import ThaiCNNPredictor
from preprocessing import _to_pil_image


# ============================================================
# Paths & Model
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "thai_cnn_classifier.pt"
TEST_IMAGES_DIR = BASE_DIR / "test_images"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print(f"Loading model from:\n{MODEL_PATH}")
predictor = ThaiCNNPredictor(model_path=MODEL_PATH)


# ============================================================
# Examples
# ============================================================

examples = []
if TEST_IMAGES_DIR.exists():
    examples = sorted(
        [[str(p)] for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:8]


def predict_sketch_or_image(image):
    if image is None:
        raise gr.Error("กรุณาวาดตัวอักษรไทยก่อนกดวิเคราะห์")
    if isinstance(image, dict):
        comp = image.get("composite", image.get("background", None))
        if comp is None:
            raise gr.Error("กรุณาวาดตัวอักษรไทยก่อนกดวิเคราะห์")
        image = comp
    return predictor.predict(image)


# ============================================================
# Gradio Application with Dual Input (Formal & Academic)
# ============================================================

with gr.Blocks(title="Thai Character Classifier (ก - ฮ)", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Thai Character Classification System")
    gr.Markdown(
        "ระบบจำแนกพยัญชนะภาษาไทย 44 รูป (ก - ฮ) ด้วยโครงข่ายประสาทเทียมสังวัตนาการ (Deep Convolutional Neural Network)\n"
        "Data Science Capstone Project | ทดสอบได้ทั้งลายมือเขียนสดบนกระดานดิจิทัลและไฟล์ภาพ"
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("วาดเขียนด้วยลายมือ (Drawing Canvas)"):
                    input_sketch = gr.Sketchpad(
                        type="numpy",
                        label="วาดพยัญชนะไทยด้วยลายมือ (Draw Thai Character)"
                    )
                    sketch_btn = gr.Button("วิเคราะห์ภาพที่วาด (Predict Drawing)", variant="primary", size="lg")

                with gr.TabItem("อัปโหลดรูปภาพ (Upload Image)"):
                    input_upload = gr.Image(
                        type="numpy",
                        label="อัปโหลดรูปภาพพยัญชนะไทย (Upload Image)"
                    )
                    upload_btn = gr.Button("วิเคราะห์ภาพที่อัปโหลด (Predict)", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_label = gr.Label(
                num_top_classes=5,
                label="Prediction (ผลการทำนาย 5 อันดับแรก)"
            )
            output_preview = gr.Image(
                type="numpy",
                label="Processed 28 × 28 image (ภาพหลังทำ Preprocessing)"
            )

    upload_btn.click(
        fn=predictor.predict,
        inputs=input_upload,
        outputs=[output_label, output_preview]
    )

    sketch_btn.click(
        fn=predict_sketch_or_image,
        inputs=input_sketch,
        outputs=[output_label, output_preview]
    )

    if examples:
        gr.Examples(
            examples=examples,
            inputs=input_upload,
            label="ชุดภาพตัวอย่างสำหรับทดสอบ (Sample Test Images)"
        )


if __name__ == "__main__":
    app.launch(share=False)
