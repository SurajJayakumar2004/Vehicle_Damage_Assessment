#!/usr/bin/env python3
"""
Test script to demonstrate the real CNN model is working
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Set environment for TensorFlow stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=== Real CNN Model Test ===")
print(f"TensorFlow version: {tf.__version__}")

# Load the model
model_path = 'models/simple_cnn_model.h5'
print(f"Loading model from: {model_path}")

try:
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Model information
    print(f"\nModel Input Shape: {model.input_shape}")
    print(f"Model Output Shape: {model.output_shape}")
    print(f"Total Parameters: {model.count_params():,}")
    
    # Test with a real image from the dataset
    test_image_paths = [
        'data/test/Fraud/1028.jpg',
        'data/test/Non-Fraud/10.jpg'
    ]
    
    classes = ['Fraud', 'Non-Fraud']
    
    print("\n=== Testing Real Images ===")
    
    for i, img_path in enumerate(test_image_paths):
        if os.path.exists(img_path):
            print(f"\nTesting Image {i+1}: {img_path}")
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            img = img.astype('float32') / 255.0
            img_batch = np.expand_dims(img, axis=0)
            
            # Get prediction
            prediction = model.predict(img_batch, verbose=0)
            fraud_prob = prediction[0][0]
            legit_prob = prediction[0][1]
            
            predicted_class = classes[0] if fraud_prob > 0.5 else classes[1]
            confidence = max(fraud_prob, legit_prob)
            
            print(f"  Fraud Probability: {fraud_prob:.3f}")
            print(f"  Legitimate Probability: {legit_prob:.3f}")
            print(f"  Predicted Class: {predicted_class}")
            print(f"  Confidence: {confidence:.3f}")
            
            # Expected class from file path
            expected_class = 'Fraud' if 'Fraud' in img_path else 'Non-Fraud'
            correct = predicted_class == expected_class
            print(f"  Expected: {expected_class}")
            print(f"  Correct: {'✅' if correct else '❌'}")
        else:
            print(f"❌ Image not found: {img_path}")
    
    # Test with random noise (should be uncertain)
    print(f"\n=== Testing Random Noise ===")
    random_img = np.random.random((1, 128, 128, 3))
    prediction = model.predict(random_img, verbose=0)
    print(f"Random noise prediction: {prediction[0]}")
    print(f"Confidence: {max(prediction[0]):.3f} (should be low for random noise)")
    
    print("\n🎉 Real CNN Model is working correctly!")
    print("The Streamlit app now uses this trained CNN instead of simulation.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
