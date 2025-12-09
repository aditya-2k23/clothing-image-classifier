import numpy as np
from PIL import Image, ImageOps
import io
import base64

def preprocess_image(file):
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
