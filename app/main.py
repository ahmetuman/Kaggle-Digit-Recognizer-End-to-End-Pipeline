from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from pathlib import Path
from pydantic import BaseModel
from typing import List
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: List[float]

class ImageRequest(BaseModel):
    image: str  # base64 encoded image

app = FastAPI(
    title="Digit Recognition API",
    description="API for recognizing handwritten digits using a CNN model trained on MNIST dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variable for our model
model = None

def get_model_path():
    """Get the model path from environment variable or use default."""
    # First, try to get from environment variable
    model_path = os.getenv('MODEL_PATH')
    if model_path:
        logger.info(f"Using model path from environment: {model_path}")
        return model_path
    
    # If not set, use default path relative to project root
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent
    default_path = str(project_root / 'models' / 'model.weights.h5')
    logger.info(f"Using default model path: {default_path}")
    return default_path

def ensure_model_loaded():
    """Ensure the model is loaded before making predictions."""
    global model
    if model is None:
        try:
            model_path = get_model_path()
            logger.info(f"Loading model from: {model_path}")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                error_msg = f"Model file not found at: {model_path}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            try:
                # Create a new model with the same architecture
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
                
                model.load_weights(model_path)
                logger.info("Model loaded successfully!")
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Unexpected error while loading model: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

def preprocess_image(image_bytes):
    """
    Preprocess the image for model prediction.
    
    Args:
        image_bytes: Raw bytes of the image
        
    Returns:
        numpy array: Preprocessed image ready for model input
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28 pixels
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    except Exception as e:
        error_msg = f"Failed to process image: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    try:
        ensure_model_loaded()
        return {
            "message": "Digit Recognition API is running",
            "model_loaded": True
        }
    except HTTPException:
        return {
            "message": "Digit Recognition API is running",
            "model_loaded": False
        }

@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    """Predict the digit in the uploaded image."""
    # Ensure model is loaded
    ensure_model_loaded()
    
    try:
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.image)
        except Exception as e:
            error_msg = f"Invalid base64 image data: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Preprocess the image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        try:
            predictions = model.predict(processed_image, verbose=0)
            predicted_digit = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            probabilities = [float(p) for p in predictions[0]]
            
            logger.info(f"Prediction successful: digit={predicted_digit}, confidence={confidence:.4f}")
            
            return PredictionResponse(
                predicted_digit=predicted_digit,
                confidence=confidence,
                probabilities=probabilities
            )
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 