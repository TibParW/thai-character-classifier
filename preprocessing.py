from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_SIZE = (28, 28)
INNER_BOX_SIZE = 20  # Leaves a 4-pixel border padding, standard for MNIST


def _to_pil_image(image):
    """
    Convert different input types into a PIL grayscale-compatible image.

    Supported inputs:
        - file path (str or Path)
        - PIL.Image
        - NumPy array
        - Gradio Sketchpad dictionary {'composite': ..., 'background': ...}
    """
    # Dictionary from Gradio Sketchpad/ImageEditor
    if isinstance(image, dict):
        if "composite" in image and image["composite"] is not None:
            image = image["composite"]
        elif "image" in image and image["image"] is not None:
            image = image["image"]
        elif "background" in image and image["background"] is not None:
            image = image["background"]
        else:
            raise ValueError("Invalid dictionary image format from Gradio component.")

    # File path
    if isinstance(image, (str, Path)):
        return Image.open(image)

    # PIL image
    if isinstance(image, Image.Image):
        return image

    # NumPy array, such as Gradio input
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)

        # Gradio or other libraries may supply float images
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr)
            if arr.max() <= 1.0:
                arr = arr * 255.0

        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Handle RGBA/RGB or 2D Grayscale
        if arr.ndim == 3 and arr.shape[2] == 4:
            # If transparent background, composite onto white background
            alpha = arr[:, :, 3] / 255.0
            rgb = arr[:, :, :3]
            white_bg = np.ones_like(rgb) * 255
            composited = (rgb * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
            return Image.fromarray(composited).convert("L")

        return Image.fromarray(arr)

    raise TypeError(f"Unsupported image type: {type(image)}")


def preprocess_image(image, target_size=IMAGE_SIZE):
    """
    Preprocess one Thai character image for model inference/training.

    Processing steps:
        1. Convert to PIL & Grayscale
        2. Standardize foreground/background (stroke = high values, background = 0)
        3. Detect character bounding box and crop
        4. Resize maintaining aspect ratio into centered 20x20 box inside 28x28
        5. Normalize pixel values to [0, 1]

    Parameters
    ----------
    image:
        File path, PIL image, NumPy array, or Gradio dictionary.

    Returns
    -------
    numpy.ndarray
        Array with shape (28, 28) and values in range [0, 1].
    """
    pil_img = _to_pil_image(image)

    # 1. Convert to grayscale
    gray = pil_img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    # 2. Standardize foreground/background:
    # A high mean (>127) indicates white background with black ink.
    # Invert so ink is bright (>0) and background is dark (0).
    if arr.mean() > 127:
        arr = 255.0 - arr

    # 3. Detect bounding box of the ink pixels
    threshold = 30.0
    y_indices, x_indices = np.where(arr > threshold)

    if len(y_indices) > 0 and len(x_indices) > 0:
        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()
        # Crop to character content
        cropped = arr[ymin : ymax + 1, xmin : xmax + 1]
    else:
        # Fallback if image is blank or below threshold
        cropped = arr

    # 4. Resize maintaining aspect ratio
    h, w = cropped.shape
    cropped_pil = Image.fromarray(np.clip(cropped, 0, 255).astype(np.uint8))

    if h > w:
        new_h = INNER_BOX_SIZE
        new_w = max(1, int(round(w * (INNER_BOX_SIZE / h))))
    else:
        new_w = INNER_BOX_SIZE
        new_h = max(1, int(round(h * (INNER_BOX_SIZE / w))))

    resized = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create target canvas (28x28 black background) and paste at center
    canvas = Image.new("L", target_size, color=0)
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    # 5. Normalize to [0, 1]
    final_arr = np.asarray(canvas, dtype=np.float64) / 255.0

    return final_arr


def image_to_features(image):
    """
    Convert an image into a 784-element feature vector.

    28 x 28 -> 784
    """
    processed = preprocess_image(image)
    return processed.reshape(-1)
