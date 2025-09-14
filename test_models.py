#!/usr/bin/env python3
"""
Model Loading Test Script
Tests if the fraud detection model can be loaded correctly from the organized structure.
"""

import os
import sys
import tensorflow as tf
import numpy as np

def test_model_loading():
    """Test loading the production model and running a simple prediction"""
    
    print("🔍 Testing Model Loading...")
    print("=" * 50)
    
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Project root: {project_root}")
    
    # Define model paths
    model_path = os.path.join(project_root, "models", "production", "fraud_detector_optimized.h5")
    threshold_path = os.path.join(project_root, "models", "production", "optimal_threshold.txt")
    
    print(f"Model path: {model_path}")
    print(f"Threshold path: {threshold_path}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
        
    if not os.path.exists(threshold_path):
        print(f"❌ Threshold file not found: {threshold_path}")
        return False
    
    try:
        # Load model
        print("\n📥 Loading model...")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Load threshold
        print("\n📥 Loading threshold...")
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        print(f"✅ Threshold loaded: {threshold}")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        test_input = np.random.rand(1, 256, 256, 3).astype('float32')
        prediction = model.predict(test_input, verbose=0)
        probability = float(prediction[0][0])
        
        print(f"   Test input shape: {test_input.shape}")
        print(f"   Prediction probability: {probability:.6f}")
        print(f"   Above threshold ({threshold}): {'Yes' if probability >= threshold else 'No'}")
        
        print("\n🎉 All tests passed! Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def test_backup_model():
    """Test loading the backup ResNet50 model"""
    
    print("\n🔍 Testing Backup Model Loading...")
    print("=" * 50)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "models", "backup", "fraud_detector_resnet50_optimized.keras")
    threshold_path = os.path.join(project_root, "models", "backup", "resnet50_optimal_threshold.txt")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Backup model not found: {model_path}")
        return False
        
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Backup ResNet50 model loaded successfully!")
        
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        print(f"   Threshold: {threshold}")
        
        return True
    except Exception as e:
        print(f"❌ Backup model error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Vehicle Damage Assessment - Model Loading Test")
    print("=" * 60)
    
    # Test primary model
    primary_success = test_model_loading()
    
    # Test backup model
    backup_success = test_backup_model()
    
    # Summary
    print("\n📊 TEST SUMMARY:")
    print("=" * 30)
    print(f"Primary Model (EfficientNet): {'✅ PASS' if primary_success else '❌ FAIL'}")
    print(f"Backup Model (ResNet50):     {'✅ PASS' if backup_success else '❌ FAIL'}")
    
    if primary_success:
        print("\n🎉 Production system is ready to use!")
        print("   Run: streamlit run app/streamlit_app.py")
    else:
        print("\n⚠️ Please check model files and paths.")
    
    return primary_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)