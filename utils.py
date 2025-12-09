import numpy as np
from PIL import Image, ImageOps
import io
import base64

def preprocess_image(file):
    """
    Preprocesses an input image file for model prediction.

    Loads the image, applies autocontrast and inversion, resizes to 28x28 pixels,
    normalizes pixel values, and returns both the processed image as a numpy array
    suitable for model input and a base64-encoded PNG string of the processed image.

    Parameters:
        file: A file-like object or path to the image file to preprocess.

    Returns:
        tuple:
            arr (np.ndarray): The processed image as a numpy array of shape (1, 28, 28, 1),
                normalized to [0, 1] for model input.
            img_bytes (str): Base64-encoded PNG string of the processed 28x28 image.
    """
    # Load
    img = Image.open(file).convert("L")

    # Autocontrast + cleanup
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    img = ImageOps.invert(img)

    # Resize
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # For model
    arr = np.array(img)
    arr = 255 - arr
    arr = arr / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    # Convert processed 28×28 image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return arr, img_bytes
