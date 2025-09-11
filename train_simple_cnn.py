#!/usr/bin/env python3
"""
Simple CNN Training Script for Vehicle Damage Fraud Detection
Creates a working TensorFlow 2.15.0 compatible model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

# Set environment variables to prevent threading issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=== Vehicle Damage Fraud Detection CNN Training ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Model configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
CLASSES = ['Fraud', 'Non-Fraud']
NUM_CLASSES = len(CLASSES)

# Data paths
TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
MODEL_SAVE_PATH = 'models/simple_cnn_model.h5'

def load_and_preprocess_data(data_path, img_size=IMG_SIZE):
    """Load images and labels from directory structure"""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found!")
            continue
            
        print(f"Loading {class_name} images from {class_path}...")
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for i, img_file in enumerate(image_files):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(image_files)} images...")
                
            try:
                img_path = os.path.join(class_path, img_file)
                
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                img = img.astype('float32') / 255.0
                
                images.append(img)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        print(f"Loaded {len([l for l in labels if l == class_idx])} {class_name} images")
    
    return np.array(images), np.array(labels)

def create_simple_cnn_model(input_shape, num_classes):
    """Create a simple but effective CNN model"""
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', name='fc1'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

def main():
    """Main training function"""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load training data
    print("\n=== Loading Training Data ===")
    X_train, y_train = load_and_preprocess_data(TRAIN_PATH)
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Load test data
    print("\n=== Loading Test Data ===")
    X_test, y_test = load_and_preprocess_data(TEST_PATH)
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ No data loaded! Check your data directories.")
        return
    
    # Convert labels to categorical
    y_train_categorical = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_categorical = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    # Create model
    print("\n=== Creating CNN Model ===")
    model = create_simple_cnn_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\n=== Training CNN Model ===")
    history = model.fit(
        X_train, y_train_categorical,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_categorical),
        verbose=1
    )
    
    # Evaluate model
    print("\n=== Evaluating Model ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate predictions for detailed evaluation
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    # Save model
    print(f"\n=== Saving Model to {MODEL_SAVE_PATH} ===")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully!")
    
    # Test loading the saved model
    print("\n=== Testing Model Loading ===")
    try:
        loaded_model = keras.models.load_model(MODEL_SAVE_PATH)
        
        # Test prediction with loaded model
        test_pred = loaded_model.predict(X_test[:1], verbose=0)
        print("✅ Model loads and predicts successfully!")
        print(f"Sample prediction: {test_pred[0]}")
        
        # Save additional model info
        model_info = {
            'input_shape': (IMG_SIZE, IMG_SIZE, 3),
            'classes': CLASSES,
            'num_classes': NUM_CLASSES,
            'test_accuracy': float(test_accuracy),
            'training_date': datetime.now().isoformat()
        }
        
        import json
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ Model info saved!")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        print("✅ Training history plot saved!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
    
    print("\n🎉 Training completed successfully!")
    print(f"Model saved at: {MODEL_SAVE_PATH}")
    print(f"Test accuracy: {test_accuracy:.1%}")

if __name__ == "__main__":
    main()
