from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Digit Recognition API",
    description="API for recognizing handwritten digits using a CNN model trained on MNIST dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

model = None

def get_model_path():
    model_path = os.getenv('MODEL_PATH')
    if model_path:
        return model_path
    
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent
    return str(project_root / 'models' / 'model.h5')

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model
    try:
        model_path = get_model_path()
        print(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            model = None
            return
            
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image for model prediction.
    """
    image = image.convert('L')
    
    image = image.resize((28, 28))
    
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {
        "message": "Digit Recognition API is running",
        "model_loaded": model is not None
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])
        
        return {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "probabilities": predictions[0].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 