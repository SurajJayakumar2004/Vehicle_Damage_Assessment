#!/usr/bin/env python3
"""
Quick Training Script for Vehicle Damage CNN
This script trains a basic CNN model for fraud detection
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import glob
from sklearn.utils import shuffle

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5  # Quick training for demonstration
CLASSES = ['Fraud', 'Non-Fraud']

def load_data(data_path, img_size, classes):
    """Load and preprocess training data"""
    print("Loading training data...")
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        print(f"Loading {class_name} images...")
        
        # Get all image files
        image_files = glob.glob(os.path.join(class_path, '*.[jp][pg]*'))  # jpg, jpeg, png
        
        # Limit to first 200 images per class for quick training
        image_files = image_files[:200]
        
        for img_file in image_files:
            try:
                # Load and preprocess image
                img = cv2.imread(img_file)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype('float32') / 255.0
                    
                    images.append(img)
                    labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
    
    print(f"Loaded {len(images)} images total")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = keras.utils.to_categorical(labels, len(classes))
    
    # Shuffle data
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

def create_model(input_shape, num_classes):
    """Create a simple CNN model"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("🚗 Vehicle Damage CNN Training Script")
    print("=" * 50)
    
    # Check if data exists
    train_path = 'data/train'
    if not os.path.exists(train_path):
        print(f"❌ Training data not found at {train_path}")
        return False
    
    # Load data
    try:
        X, y = load_data(train_path, IMG_SIZE, CLASSES)
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1:]} shape")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    model = create_model((IMG_SIZE, IMG_SIZE, 3), len(CLASSES))
    print("✅ Model created")
    model.summary()
    
    # Train model
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Evaluate model
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"📊 Validation Accuracy: {val_acc:.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    # Save in both formats
    model.save('models/vehicle_damage_model.keras')
    model.save('models/vehicle_damage_cnn_model.h5')
    
    print("✅ Model saved to models/ directory")
    print("🎉 Training completed successfully!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🔗 You can now use the Streamlit app with the trained model!")
    else:
        print("\n❌ Training failed. Please check the errors above.")
