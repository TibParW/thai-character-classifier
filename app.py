from pathlib import Path

import gradio as gr
import torch

from cnn_model import ThaiCNNPredictor


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "thai_cnn_classifier.pt"
)

TEST_IMAGES_DIR = BASE_DIR / "test_images"


# ============================================================
# Load model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "\n"
        "Model file not found:\n"
        f"{MODEL_PATH}\n\n"
        "Run train_cnn.py first."
    )

print(
    f"Loading model from:\n{MODEL_PATH}"
)

predictor = ThaiCNNPredictor(
    model_path=MODEL_PATH
)


# ============================================================
# Examples
# ============================================================

examples = []
if TEST_IMAGES_DIR.exists():
    examples = sorted(
        [[str(p)] for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:8]


# ============================================================
# Gradio application
# ============================================================

app = gr.Interface(
    fn=predictor.predict,

    inputs=gr.Image(
        type="numpy",
        label="Upload a Thai character image (อัปโหลดรูปพยัญชนะไทย)"
    ),

    outputs=[
        gr.Label(
            num_top_classes=5,
            label="Prediction (ผลการทำนาย 5 อันดับแรก)"
        ),

        gr.Image(
            type="numpy",
            label="Processed 28 × 28 image (ภาพหลังทำ Preprocessing)"
        )
    ],

    title=(
        "Thai Character Classifier (ก - ฮ)"
    ),

    description=(
        "Upload an image containing one Thai consonant from ก to ฮ. "
        "The application preprocesses the image and uses a trained Deep CNN model "
        "(99.14% accuracy on 22,000 samples) to classify the character."
    ),

    examples=examples if examples else None,
)


# ============================================================
# Start application
# ============================================================

if __name__ == "__main__":
    app.launch(share=False)
