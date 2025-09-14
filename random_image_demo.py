"""
Quick Filter Visualization Demo
==============================

This script will automatically pick a random image from your test dataset
and show what your CNN filters learn when processing it.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import random
import glob

def load_model_or_create_sample():
    """Load the actual model or create a sample one"""
    model_paths = [
        "vehicle_damage_fraud_model.h5",
        "model/vehicle_damage_fraud_model.h5",
        "models/vehicle_damage_fraud_model.h5"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                print(f"✅ Loaded actual model from: {path}")
                return model, True
            except Exception as e:
                print(f"⚠️ Error loading {path}: {str(e)}")
                continue
    
    # Create sample model
    print("⚠️ Creating sample CNN model for demonstration")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), name='conv2d_1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_1'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_2'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_3'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    return model, False

def get_random_test_image():
    """Get a random image from the test dataset"""
    # Look for test images in your dataset
    test_paths = [
        "data/test/Fraud/*.jpg",
        "data/test/Non-Fraud/*.jpg",
        "data/test/Fraud/*.jpeg",
        "data/test/Non-Fraud/*.jpeg",
        "data/test/Fraud/*.png",
        "data/test/Non-Fraud/*.png"
    ]
    
    all_images = []
    for pattern in test_paths:
        all_images.extend(glob.glob(pattern))
    
    if all_images:
        selected_image = random.choice(all_images)
        category = "Fraud" if "Fraud" in selected_image else "Non-Fraud"
        print(f"📸 Selected random image: {os.path.basename(selected_image)} ({category})")
        return selected_image, category
    else:
        print("❌ No test images found in data/test/ directory")
        return None, None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    try:
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, image
    except Exception as e:
        print(f"❌ Error preprocessing image: {str(e)}")
        return None, None

def visualize_filters(model, layer_name, num_filters=16):
    """Visualize learned filters from a specific layer"""
    try:
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()[0]
        
        # Normalize weights
        weights_norm = (weights - weights.mean()) / (weights.std() + 1e-8)
        
        # Setup plot
        cols = 4
        rows = (num_filters + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        fig.suptitle(f'🔍 Learned Filters in {layer_name}', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        num_filters = min(num_filters, weights.shape[-1])
        
        for i in range(num_filters):
            row = i // cols
            col = i % cols
            
            # Get filter (use first input channel)
            filter_weights = weights_norm[:, :, 0, i]
            
            # Plot
            im = axes[row, col].imshow(filter_weights, cmap='RdBu', aspect='auto')
            axes[row, col].set_title(f'Filter {i+1}', fontsize=10)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for i in range(num_filters, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error visualizing filters: {str(e)}")

def create_feature_maps(model, image_array, layer_name):
    """Create feature maps for the image"""
    try:
        # Create intermediate model
        intermediate_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get feature maps
        feature_maps = intermediate_model.predict(image_array, verbose=0)
        return feature_maps[0]  # Remove batch dimension
        
    except Exception as e:
        print(f"❌ Error creating feature maps: {str(e)}")
        return None

def visualize_feature_maps(original_image, feature_maps, layer_name, num_maps=16):
    """Visualize feature maps alongside the original image"""
    if feature_maps is None:
        return
    
    try:
        num_maps = min(num_maps, feature_maps.shape[-1])
        cols = 4
        rows = (num_maps + cols - 1) // cols
        
        # Create figure with extra space for original image
        fig, axes = plt.subplots(rows, cols + 1, figsize=(20, rows * 4))
        fig.suptitle(f'🖼️ Feature Maps from {layer_name}', fontsize=16, fontweight='bold')
        
        # Show original image in first column
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('🖼️ Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Add image info
        if hasattr(original_image, 'size'):
            axes[0, 0].text(0.5, -0.1, f'Size: {original_image.size}', 
                           transform=axes[0, 0].transAxes, ha='center', fontsize=10)
        
        # Hide other cells in first column
        for i in range(1, rows):
            axes[i, 0].axis('off')
        
        # Show feature maps
        for i in range(num_maps):
            row = i // cols
            col = i % cols + 1
            
            feature_map = feature_maps[:, :, i]
            im = axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'Feature Map {i+1}', fontsize=10)
            axes[row, col].axis('off')
            
            # Add activation info
            max_activation = np.max(feature_map)
            axes[row, col].text(0.5, -0.1, f'Max: {max_activation:.2f}', 
                               transform=axes[row, col].transAxes, ha='center', fontsize=8)
            
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for i in range(num_maps, rows * cols):
            row = i // cols
            col = i % cols + 1
            if col < axes.shape[1]:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error visualizing feature maps: {str(e)}")

def analyze_model_prediction(model, image_array, image_path, is_real_model):
    """Analyze what the model predicts for this image"""
    try:
        if is_real_model:
            # Make prediction
            prediction = model.predict(image_array, verbose=0)[0][0]
            
            # Interpret prediction
            is_fraud = prediction > 0.5
            confidence = prediction if is_fraud else 1 - prediction
            
            print(f"\n🎯 Model Prediction Analysis:")
            print(f"   Image: {os.path.basename(image_path)}")
            print(f"   Prediction: {'🚨 FRAUD' if is_fraud else '✅ LEGITIMATE'}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Raw Score: {prediction:.4f}")
            
            # Determine if prediction matches file location
            actual_category = "Fraud" if "Fraud" in image_path else "Non-Fraud"
            predicted_category = "Fraud" if is_fraud else "Non-Fraud"
            
            if actual_category == predicted_category:
                print(f"   Result: ✅ CORRECT prediction!")
            else:
                print(f"   Result: ❌ Incorrect (should be {actual_category})")
                
        else:
            print(f"\n🎯 Using Sample Model:")
            print(f"   (Predictions are random - not trained on your data)")
            
    except Exception as e:
        print(f"❌ Error analyzing prediction: {str(e)}")

def main():
    """Main demonstration function"""
    print("🔍 CNN Filter Visualization with Random Image")
    print("=" * 60)
    
    # Load model
    model, is_real_model = load_model_or_create_sample()
    
    # Get random test image
    image_path, category = get_random_test_image()
    
    if image_path is None:
        print("❌ No test images found. Please add images to data/test/Fraud/ or data/test/Non-Fraud/")
        return
    
    # Preprocess image
    image_array, original_image = preprocess_image(image_path)
    
    if image_array is None:
        return
    
    print(f"📊 Image Info:")
    print(f"   Original Category: {category}")
    print(f"   Processed Shape: {image_array.shape}")
    
    # Analyze prediction
    analyze_model_prediction(model, image_array, image_path, is_real_model)
    
    # Get convolutional layers
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer.name)
    
    print(f"\n🏗️ Found {len(conv_layers)} convolutional layers: {conv_layers}")
    
    # Visualize filters and feature maps for each conv layer
    for i, layer_name in enumerate(conv_layers):
        print(f"\n📊 Analyzing Layer {i+1}/{len(conv_layers)}: {layer_name}")
        
        # Show filters
        print("   🔍 Visualizing learned filters...")
        visualize_filters(model, layer_name, num_filters=16)
        
        # Show feature maps
        print("   🖼️ Creating feature maps...")
        feature_maps = create_feature_maps(model, image_array, layer_name)
        visualize_feature_maps(original_image, feature_maps, layer_name, num_maps=16)
        
        # Show layer stats
        try:
            layer = model.get_layer(layer_name)
            weights = layer.get_weights()[0]
            print(f"   📈 Layer Stats: Shape={weights.shape}, Mean={np.mean(weights):.4f}, Std={np.std(weights):.4f}")
        except:
            pass
    
    print(f"\n✅ Analysis Complete!")
    print(f"🎯 Key Insights:")
    print(f"   • Early layers detect edges and simple features")
    print(f"   • Later layers detect complex patterns")
    print(f"   • Feature maps show where filters activate strongly")
    print(f"   • Bright areas in feature maps = strong activation")
    
    if is_real_model:
        print(f"   • This analysis used your actual trained model!")
    else:
        print(f"   • Train your model to see real fraud detection patterns!")

if __name__ == "__main__":
    main()