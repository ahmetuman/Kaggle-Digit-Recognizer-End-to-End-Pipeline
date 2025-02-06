import pytest
from fastapi import FastAPI
from PIL import Image
import numpy as np
import io

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "model_loaded" in response.json()

def test_prediction_endpoint_valid_image(client, test_image):
    """Test prediction endpoint with a valid image."""
    response = client.post(
        "/predict/",
        files={"file": ("test_image.png", test_image, "image/png")}
    )
    assert response.status_code == 200
    
    # Check response structure
    json_response = response.json()
    assert "predicted_digit" in json_response
    assert "confidence" in json_response
    assert "probabilities" in json_response
    
    # Validate response data types
    assert isinstance(json_response["predicted_digit"], int)
    assert isinstance(json_response["confidence"], float)
    assert isinstance(json_response["probabilities"], list)
    assert len(json_response["probabilities"]) == 10  # One probability per digit (0-9)

def test_prediction_endpoint_invalid_image(client, invalid_image):
    """Test prediction endpoint with invalid image data."""
    response = client.post(
        "/predict/",
        files={"file": ("invalid.png", invalid_image, "image/png")}
    )
    assert response.status_code == 500
    assert "detail" in response.json()

def test_prediction_endpoint_no_file(client):
    """Test prediction endpoint without providing a file."""
    response = client.post("/predict/")
    assert response.status_code == 422  # Unprocessable Entity

def test_image_preprocessing(client, test_image):
    """Test that image preprocessing works correctly."""
    # Get the original image
    img = Image.open(test_image)
    img_array = np.array(img)
    
    # Make prediction
    test_image.seek(0)  # Reset file pointer
    response = client.post(
        "/predict/",
        files={"file": ("test_image.png", test_image, "image/png")}
    )
    
    assert response.status_code == 200
    
    # Verify that preprocessing didn't fail
    assert "predicted_digit" in response.json()

def test_model_confidence(client, test_image):
    """Test that model confidence scores are valid."""
    response = client.post(
        "/predict/",
        files={"file": ("test_image.png", test_image, "image/png")}
    )
    
    assert response.status_code == 200
    json_response = response.json()
    
    # Check confidence score is between 0 and 1
    assert 0 <= json_response["confidence"] <= 1
    
    # Check probabilities sum to approximately 1
    assert abs(sum(json_response["probabilities"]) - 1.0) < 1e-6

def test_concurrent_requests(client, test_image):
    """Test handling multiple requests in quick succession."""
    n_requests = 5
    responses = []
    
    for _ in range(n_requests):
        test_image.seek(0)
        response = client.post(
            "/predict/",
            files={"file": ("test_image.png", test_image, "image/png")}
        )
        responses.append(response)
    
    # Check all requests were successful
    for response in responses:
        assert response.status_code == 200
        assert "predicted_digit" in response.json() 