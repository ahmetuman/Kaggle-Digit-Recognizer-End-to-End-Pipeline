import pytest
from fastapi import FastAPI
from PIL import Image
import numpy as np
import io
import base64

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "model_loaded" in response.json()

def test_prediction_endpoint_valid_image(client, test_image):
    """Test prediction endpoint with a valid image."""
    test_image.seek(0)
    image_data = base64.b64encode(test_image.read()).decode('utf-8')
    
    response = client.post(
        "/predict/",
        json={"image": image_data}
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
    # Convert invalid data to base64
    invalid_image.seek(0)
    image_data = base64.b64encode(invalid_image.read()).decode('utf-8')
    
    response = client.post(
        "/predict/",
        json={"image": image_data}
    )
    assert response.status_code == 500
    assert "detail" in response.json()

def test_prediction_endpoint_invalid_base64(client):
    """Test prediction endpoint with invalid base64 data."""
    response = client.post(
        "/predict/",
        json={"image": "invalid_base64_data"}
    )
    assert response.status_code == 400  # Bad Request
    assert "detail" in response.json()

def test_prediction_endpoint_no_data(client):
    """Test prediction endpoint without providing data."""
    response = client.post("/predict/")
    assert response.status_code == 422  # Unprocessable Entity

def test_image_preprocessing(client, test_image):
    """Test that image preprocessing works correctly."""
    test_image.seek(0)
    image_data = base64.b64encode(test_image.read()).decode('utf-8')
    
    response = client.post(
        "/predict/",
        json={"image": image_data}
    )
    
    assert response.status_code == 200
    assert "predicted_digit" in response.json()

def test_model_confidence(client, test_image):
    """Test that model confidence scores are valid."""
    # Convert image to base64
    test_image.seek(0)
    image_data = base64.b64encode(test_image.read()).decode('utf-8')
    
    response = client.post(
        "/predict/",
        json={"image": image_data}
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
    
    # Convert image to base64 once
    test_image.seek(0)
    image_data = base64.b64encode(test_image.read()).decode('utf-8')
    
    for _ in range(n_requests):
        response = client.post(
            "/predict/",
            json={"image": image_data}
        )
        responses.append(response)
    
    # Check all requests were successful
    for response in responses:
        assert response.status_code == 200
        assert "predicted_digit" in response.json() 