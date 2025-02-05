import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_api_health():
    """Test the API health check endpoint."""
    response = requests.get("http://localhost:8000/")
    print("Health Check Response:", response.json())
    return response.status_code == 200

def test_prediction(image_path):
    """Test the prediction endpoint with an image."""
    files = {'file': open(image_path, 'rb')}
    response = requests.post("http://localhost:8000/predict/", files=files)
    print("Prediction Response:", response.json())
    return response.status_code == 200

def create_test_image():
    """Create a test image (digit 5) for testing."""
    # Create a 28x28 black image
    img = np.zeros((28, 28), dtype=np.uint8)
    
    # Draw a simple "5"
    img[5:20, 5:20] = 255  # Draw a white rectangle
    
    # Save the image
    image = Image.fromarray(img)
    image.save("test_digit.png")
    
    # Display the image
    plt.imshow(img, cmap='gray')
    plt.title("Test Digit Image")
    plt.show()
    
    return "test_digit.png"

if __name__ == "__main__":
    print("Creating test image...")
    test_image_path = create_test_image()
    
    print("\nTesting API health...")
    if test_api_health():
        print("Health check passed!")
    
    print("\nTesting prediction...")
    if test_prediction(test_image_path):
        print("Prediction test passed!") 