import pytest
from fastapi.testclient import TestClient
from app.main import app
import numpy as np
from PIL import Image
import io
import os

@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)

@pytest.fixture
def test_image():
    """Create a test image for predictions."""
    # Create a 28x28 test image (similar to MNIST format)
    img = np.zeros((28, 28), dtype=np.uint8)
    img[5:20, 5:20] = 255  # Create a simple square pattern
    
    # Convert to PIL Image
    image = Image.fromarray(img)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

@pytest.fixture
def invalid_image():
    """Create an invalid image for testing error handling."""
    return io.BytesIO(b"invalid image content") 