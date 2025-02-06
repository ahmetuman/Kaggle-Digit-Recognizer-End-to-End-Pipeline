import numpy as np
from PIL import Image
import io
import base64
import json

# Create a 28x28 test image (similar to MNIST format)
img = np.zeros((28, 28), dtype=np.uint8)
img[5:20, 5:20] = 255  # Create a simple square pattern

image = Image.fromarray(img)

img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

base64_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

json_data = {
    "image": base64_str
}

# Save to file
with open('tests/test_image.json', 'w') as f:
    json.dump(json_data, f) 