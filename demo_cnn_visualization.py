#!/usr/bin/env python3
"""
Quick Demo of CNN Visualization Features
Shows layer outputs, edge detection, and attention maps
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Set environment
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=== CNN Visualization Demo ===")

# Load existing model
model_path = 'models/simple_cnn_model.h5'
try:
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except:
    print("❌ Could not load model. Please run training first.")
    exit(1)

# Load a test image
test_images = [
    'data/test/Fraud/1028.jpg',
    'data/test/Non-Fraud/10.jpg'
]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\n🔍 Analyzing: {img_path}")
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img_normalized = img.astype('float32') / 255.0
        
        # Get prediction
        prediction = model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0]
        predicted_class = 'Fraud' if prediction[0] > 0.5 else 'Non-Fraud'
        confidence = max(prediction)
        
        print(f"Prediction: {predicted_class} (Confidence: {confidence:.3f})")
        print(f"Fraud Prob: {prediction[0]:.3f}, Non-Fraud Prob: {prediction[1]:.3f}")
        
        # 1. Show layer outputs
        print("\n1. Layer-by-Layer Analysis:")
        conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
        
        fig, axes = plt.subplots(len(conv_layers), 4, figsize=(16, len(conv_layers) * 3))
        
        for layer_idx, layer in enumerate(conv_layers[:4]):  # First 4 conv layers
            # Get layer output
            layer_model = keras.Model(inputs=model.input, outputs=layer.output)
            layer_output = layer_model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0]
            
            print(f"   Layer {layer_idx+1} ({layer.name}): Output shape {layer_output.shape}")
            
            # Show first 4 feature maps
            for i in range(min(4, layer_output.shape[-1])):
                ax = axes[layer_idx, i] if len(conv_layers) > 1 else axes[i]
                ax.imshow(layer_output[:, :, i], cmap='viridis')
                ax.set_title(f'{layer.name}\nFilter {i+1}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'layer_analysis_{os.path.basename(img_path)}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Edge detection comparison
        print("\n2. Edge Detection Analysis:")
        
        # Traditional edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # CNN first layer output
        first_conv_output = keras.Model(inputs=model.input, outputs=conv_layers[0].output).predict(
            np.expand_dims(img_normalized, axis=0), verbose=0
        )[0]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Original and traditional edges
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(edges_canny, cmap='gray')
        axes[0, 1].set_title('Canny Edge Detection')
        axes[0, 1].axis('off')
        
        # CNN learned filters (fill remaining slots in 2x4 grid)
        filter_positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        for i in range(min(6, len(filter_positions))):
            if i < first_conv_output.shape[-1]:
                row, col = filter_positions[i]
                axes[row, col].imshow(first_conv_output[:, :, i], cmap='viridis')
                axes[row, col].set_title(f'CNN Filter {i+1}')
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'edge_analysis_{os.path.basename(img_path)}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Simple attention visualization
        print("\n3. Attention Analysis:")
        
        # Get final conv layer output
        final_conv_layer = conv_layers[-1]
        final_conv_output = keras.Model(inputs=model.input, outputs=final_conv_layer.output).predict(
            np.expand_dims(img_normalized, axis=0), verbose=0
        )[0]
        
        # Create simple attention map by averaging feature maps
        attention_map = np.mean(final_conv_output, axis=-1)
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Resize to original image size
        attention_resized = cv2.resize(attention_map, (128, 128))
        
        # Create heatmap overlay
        heatmap_colored = cv2.applyColorMap(
            (attention_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
        
        # Find important regions and draw circles
        binary_mask = (attention_resized > 0.6).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circled_image = overlay.copy()
        detection_count = 0
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                detection_count += 1
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2
                radius = max(w, h) // 2 + 5
                
                # Draw green circle
                cv2.circle(circled_image, (center_x, center_y), radius, (0, 255, 0), 2)
                # Draw red bounding box
                cv2.rectangle(circled_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
                # Add number
                cv2.putText(circled_image, str(detection_count), (center_x-5, center_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        print(f"   Found {detection_count} important detection areas")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(attention_map, cmap='hot')
        axes[0, 1].set_title('CNN Attention Map')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Attention Overlay')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(circled_image)
        axes[1, 1].set_title(f'Detection Areas (Circled)\n{predicted_class} - {confidence:.1%}')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'attention_analysis_{os.path.basename(img_path)}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved for {os.path.basename(img_path)}")

print(f"\n🎉 CNN Visualization Demo Complete!")
print(f"Generated visualization files:")
print(f"  - layer_analysis_*.png: Shows how each CNN layer processes the image")
print(f"  - edge_analysis_*.png: Compares CNN edge detection vs traditional methods")
print(f"  - attention_analysis_*.png: Shows what areas the CNN focuses on with circles")
print(f"\nThese demonstrate:")
print(f"  ✅ Layer-by-layer feature learning")
print(f"  ✅ Edge detection and pattern recognition")
print(f"  ✅ Attention mapping with detection circles")
print(f"  ✅ How the CNN learns to identify fraud vs legitimate damage")
