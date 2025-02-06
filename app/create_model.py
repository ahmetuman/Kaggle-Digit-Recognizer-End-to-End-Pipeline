import tensorflow as tf
from tensorflow.keras import layers, models
import os
from pathlib import Path

def create_mnist_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, output_path):
    """Save the model weights to the specified path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save only the weights
    model.save_weights(output_path)
    print(f"Model weights saved to: {output_path}")

if __name__ == "__main__":
    # Create the model
    model = create_mnist_model()
    
    # Get the model path
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent
    model_path = str(project_root / 'models' / 'model.weights.h5')
    
    # Save the model
    save_model(model, model_path) 