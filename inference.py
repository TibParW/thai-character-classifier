import numpy as np
from PIL import Image
from preprocessing import preprocess_image
from labels import LABEL_TO_DISPLAY


class ThaiCharacterPredictor:
    """
    Prediction service for Thai Consonant Classification (44 classes).

    The model is injected through the constructor rather than
    accessed as a global variable.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, image):
        """
        Predict Thai consonant from image.

        Parameters
        ----------
        image:
            Image supplied by Gradio (numpy array, PIL image, or sketchpad dict).

        Returns
        -------
        scores:
            Dictionary of class probabilities for Gradio Label output.
        preview:
            28x28 processed preview image for display.
        """
        if image is None:
            raise ValueError("กรุณาอัปโหลดรูปภาพหรือวาดตัวอักษรไทย (Please upload an image or sketch a character)")

        # 1. Preprocess
        processed = preprocess_image(image)

        # 2. Reshape 28 x 28 -> 1 x 784
        x = processed.reshape(1, -1)

        # 3. Predict probabilities
        probabilities = self.model.predict_proba(x)[0]
        raw_classes = self.model.classes_

        # 4. Map folder class names to friendly Thai display names e.g. "ก (ก ไก่)"
        scores = {}
        for raw_label, prob in zip(raw_classes, probabilities):
            display_name = LABEL_TO_DISPLAY.get(raw_label, str(raw_label))
            scores[display_name] = float(prob)

        # 5. Preview image (enlarged to 280x280 with NEAREST for clear visible pixels)
        preview_28 = (processed * 255.0).astype(np.uint8)
        preview = np.array(Image.fromarray(preview_28).resize((280, 280), Image.Resampling.NEAREST))

        return scores, preview

    def predict_class(self, image):
        """
        Return only the top predicted Thai consonant string.
        """
        processed = preprocess_image(image)
        x = processed.reshape(1, -1)
        raw_pred = self.model.predict(x)[0]
        return LABEL_TO_DISPLAY.get(raw_pred, str(raw_pred))

    def diagnose(self, image):
        """
        Return useful diagnostics for debugging.
        """
        processed = preprocess_image(image)
        x = processed.reshape(1, -1)
        probabilities = self.model.predict_proba(x)[0]
        raw_pred = self.model.predict(x)[0]

        return {
            "shape": x.shape,
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "prediction": LABEL_TO_DISPLAY.get(raw_pred, str(raw_pred)),
            "top_probabilities": sorted(
                [
                    {"label": LABEL_TO_DISPLAY.get(l, str(l)), "prob": float(p)}
                    for l, p in zip(self.model.classes_, probabilities)
                ],
                key=lambda item: item["prob"],
                reverse=True,
            )[:5],
        }
