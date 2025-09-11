#!/usr/bin/env python3
"""
Advanced Fraud Detection App with CNN Visualization
Shows layer outputs, edge detection, and feature highlighting
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import json
from datetime import datetime
import seaborn as sns

# Configure TensorFlow to reduce warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page configuration
st.set_page_config(
    page_title="Advanced CNN Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set environment for TensorFlow
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .visualization-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    .detection-highlight {
        border: 3px solid #ff4444;
        border-radius: 10px;
        padding: 10px;
        background: rgba(255, 68, 68, 0.1);
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Configuration
IMG_SIZE = 128
CLASSES = ['Fraud', 'Non-Fraud']

@st.cache_resource
def load_advanced_model():
    """Load the advanced CNN model"""
    try:
        model_path = '../models/advanced_cnn_model.h5'
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            st.success("✅ Advanced CNN Model loaded successfully!")
            return model
        else:
            # Try the simple model as fallback
            simple_path = '../models/simple_cnn_model.h5'
            if os.path.exists(simple_path):
                model = keras.models.load_model(simple_path)
                st.warning("⚠️ Using simple CNN model (advanced model not found)")
                return model
            else:
                st.error("❌ No trained model found! Please run the training script first.")
                return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def get_layer_outputs(model, image, layer_names):
    """Get outputs from specified CNN layers"""
    layer_outputs = {}
    
    for layer_name in layer_names:
        try:
            # Create intermediate model
            intermediate_model = keras.Model(
                inputs=model.input,
                outputs=model.get_layer(layer_name).output
            )
            output = intermediate_model.predict(np.expand_dims(image, axis=0), verbose=0)
            layer_outputs[layer_name] = output[0]
        except Exception as e:
            st.warning(f"Could not get output for layer {layer_name}: {e}")
    
    return layer_outputs

def create_attention_heatmap(model, image, predicted_class_idx):
    """Create attention heatmap showing important image regions"""
    try:
        # Find last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create gradient model
        grad_model = keras.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            class_prob = predictions[:, predicted_class_idx]
        
        grads = tape.gradient(class_prob, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        for i in range(pooled_grads.shape[-1]):
            conv_outputs = conv_outputs[:, :, i:i+1] * pooled_grads[i]
        
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Error creating attention heatmap: {e}")
        return None

def highlight_detection_areas(image, heatmap, threshold=0.6):
    """Highlight important areas and draw detection circles"""
    if heatmap is None:
        return image, image
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Fix NaN and clip values to valid range
    heatmap_resized = np.nan_to_num(heatmap_resized, nan=0.0)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    
    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    overlay = cv2.addWeighted(
        (image * 255).astype(np.uint8), 
        0.7, 
        heatmap_colored, 
        0.3, 
        0
    )
    
    # Find important regions and draw circles
    binary_mask = (heatmap_resized > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circled_image = overlay.copy()
    detection_info = []
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 50:  # Filter small areas
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw green circle around detection area
            center_x, center_y = x + w//2, y + h//2
            radius = max(w, h) // 2 + 10
            cv2.circle(circled_image, (center_x, center_y), radius, (0, 255, 0), 3)
            
            # Draw bounding box
            cv2.rectangle(circled_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add detection number
            cv2.putText(circled_image, f"{i+1}", (center_x-10, center_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            detection_info.append({
                'id': i+1,
                'center': (center_x, center_y),
                'size': (w, h),
                'area': cv2.contourArea(contour),
                'importance': np.mean(heatmap_resized[y:y+h, x:x+w])
            })
    
    return circled_image, overlay, detection_info

def analyze_edge_detection(image, model):
    """Analyze edge detection in first convolutional layer"""
    try:
        # Get first conv layer output
        conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
        if not conv_layers:
            return None, None
        
        first_conv = conv_layers[0]
        conv_model = keras.Model(inputs=model.input, outputs=first_conv.output)
        conv_output = conv_model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        
        # Traditional edge detection for comparison
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges_canny = cv2.Canny(gray_image, 50, 150)
        
        return conv_output, edges_canny
    except Exception as e:
        st.error(f"Error in edge detection analysis: {e}")
        return None, None

def display_layer_visualizations(layer_outputs):
    """Display CNN layer outputs in an organized way"""
    
    layer_descriptions = {
        'conv1_edge_detection': 'Edge Detection - Identifies basic edges and textures',
        'conv2_patterns': 'Pattern Recognition - Combines edges into patterns',
        'conv3_features': 'Feature Combination - Recognizes complex shapes',
        'conv4_highlevel': 'High-level Features - Detects damage-specific features'
    }
    
    for layer_name, output in layer_outputs.items():
        with st.expander(f"🔍 {layer_name.replace('_', ' ').title()} Analysis", expanded=False):
            description = layer_descriptions.get(layer_name, 'CNN layer output')
            st.write(f"**{description}**")
            
            # Show feature maps
            num_filters = min(8, output.shape[-1])
            cols = 4
            rows = 2
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
            axes = axes.ravel()
            
            for i in range(num_filters):
                axes[i].imshow(output[:, :, i], cmap='viridis')
                axes[i].set_title(f'Filter {i+1}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_filters, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show statistics
            st.write(f"**Layer Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Output Shape", f"{output.shape}")
            with col2:
                st.metric("Active Filters", f"{int(np.sum(np.max(output, axis=(0,1)) > 0.1))}")
            with col3:
                st.metric("Max Activation", f"{float(np.max(output)):.3f}")
            with col4:
                st.metric("Mean Activation", f"{float(np.mean(output)):.3f}")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Advanced CNN Fraud Detection</h1>
        <p>Complete Layer-by-Layer Analysis with Feature Visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_advanced_model()
    
    if model is None:
        st.stop()
    
    # Sidebar with model information
    with st.sidebar:
        st.markdown("### 🤖 Model Information")
        
        try:
            # Try to load model info
            info_path = '../models/advanced_model_info.json'
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                
                st.write(f"**Training Date:** {model_info.get('training_date', 'Unknown')[:10]}")
                if 'metrics' in model_info:
                    metrics = model_info['metrics']
                    st.write(f"**Accuracy:** {metrics.get('accuracy', 0):.1%}")
                    st.write(f"**Precision:** {metrics.get('precision', 0):.1%}")
                    st.write(f"**Recall:** {metrics.get('recall', 0):.1%}")
                    st.write(f"**F1-Score:** {metrics.get('f1_score', 0):.1%}")
        except:
            st.write("Model info not available")
        
        st.markdown("### 📊 Analysis Features")
        st.write("""
        ✅ **Layer-by-Layer Visualization**
        - Edge detection filters
        - Pattern recognition
        - Feature combination
        - High-level analysis
        
        ✅ **Attention Mapping**
        - Shows focus areas
        - Highlights important regions
        - Circles detection zones
        
        ✅ **Edge Analysis**
        - CNN vs traditional methods
        - Filter comparison
        - Feature learning
        """)
    
    # Main interface
    st.markdown("### 📷 Upload Image for Complete Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a vehicle damage image",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload an image to see complete CNN analysis with layer visualizations"
    )
    
    if uploaded_file is not None:
        # Load and preprocess image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Resize and normalize for model
        image_resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
        if len(image_resized.shape) == 2:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        image_normalized = image_resized.astype('float32') / 255.0
        
        # Display original image
        st.subheader("📷 Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", width=400)
        
        # Analysis button
        if st.button("🔍 Start Complete CNN Analysis", type="primary"):
            with st.spinner("Performing deep CNN analysis..."):
                
                # 1. Get CNN Prediction
                prediction = model.predict(np.expand_dims(image_normalized, axis=0), verbose=0)[0]
                predicted_class_idx = np.argmax(prediction)
                predicted_class = CLASSES[predicted_class_idx]
                confidence = float(prediction[predicted_class_idx])
                
                # Display prediction results
                st.subheader("🎯 CNN Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Class", predicted_class)
                with col2:
                    conf_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
                    st.markdown(f'<p class="{conf_class}">Confidence: {confidence:.1%}</p>', unsafe_allow_html=True)
                with col3:
                    fraud_prob = float(prediction[0])
                    st.metric("Fraud Probability", f"{fraud_prob:.1%}")
                
                # Probability bars
                st.write("**Class Probabilities:**")
                for i, (class_name, prob) in enumerate(zip(CLASSES, prediction)):
                    st.progress(float(prob), text=f"{class_name}: {prob:.1%}")
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🧠 Layer Analysis", 
                    "🔍 Attention Map", 
                    "⚡ Edge Detection", 
                    "🎯 Detection Areas",
                    "📊 Summary"
                ])
                
                with tab1:
                    st.subheader("🧠 CNN Layer-by-Layer Analysis")
                    st.write("This shows how each layer of the CNN processes the image:")
                    
                    # Get layer outputs
                    layer_names = []
                    for layer in model.layers:
                        if isinstance(layer, keras.layers.Conv2D):
                            layer_names.append(layer.name)
                    
                    if layer_names:
                        layer_outputs = get_layer_outputs(model, image_normalized, layer_names[:4])
                        display_layer_visualizations(layer_outputs)
                    else:
                        st.warning("No convolutional layers found for visualization")
                
                with tab2:
                    st.subheader("🔍 Attention Heatmap")
                    st.write("This shows what parts of the image the CNN focuses on for its decision:")
                    
                    heatmap = create_attention_heatmap(model, image_normalized, predicted_class_idx)
                    
                    if heatmap is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Attention Heatmap**")
                            fig, ax = plt.subplots(figsize=(6, 6))
                            im = ax.imshow(heatmap, cmap='hot')
                            ax.set_title('CNN Attention Areas')
                            ax.axis('off')
                            plt.colorbar(im)
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.write("**Overlay on Original**")
                            heatmap_resized = cv2.resize(heatmap, (image_resized.shape[1], image_resized.shape[0]))
                            # Fix NaN and clip values to valid range
                            heatmap_resized = np.nan_to_num(heatmap_resized, nan=0.0)
                            heatmap_resized = np.clip(heatmap_resized, 0, 1)
                            heatmap_colored = cv2.applyColorMap(
                                (heatmap_resized * 255).astype(np.uint8), 
                                cv2.COLORMAP_JET
                            )
                            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                            overlay = cv2.addWeighted(image_resized, 0.7, heatmap_colored, 0.3, 0)
                            
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(overlay)
                            ax.set_title('Attention Overlay')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.warning("Could not generate attention heatmap")
                
                with tab3:
                    st.subheader("⚡ Edge Detection Analysis")
                    st.write("Comparison between CNN learned filters and traditional edge detection:")
                    
                    conv_output, edges_canny = analyze_edge_detection(image_normalized, model)
                    
                    if conv_output is not None and edges_canny is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Traditional Canny Edge Detection**")
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(edges_canny, cmap='gray')
                            ax.set_title('Canny Edges')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.write("**CNN Learned Edge Filters**")
                            fig, axes = plt.subplots(2, 2, figsize=(6, 6))
                            axes = axes.ravel()
                            
                            for i in range(min(4, conv_output.shape[-1])):
                                axes[i].imshow(conv_output[:, :, i], cmap='viridis')
                                axes[i].set_title(f'CNN Filter {i+1}')
                                axes[i].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        st.info("💡 The CNN learns its own edge detection filters that are specifically optimized for fraud detection!")
                
                with tab4:
                    st.subheader("🎯 Detection Areas with Circles")
                    st.write("Important regions highlighted with detection circles:")
                    
                    if heatmap is not None:
                        circled_image, overlay, detection_info = highlight_detection_areas(
                            image_normalized, heatmap, threshold=0.6
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Detection Areas Circled**")
                            fig, ax = plt.subplots(figsize=(8, 8))
                            ax.imshow(circled_image / 255.0)
                            ax.set_title('Important Areas for Decision Making')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.write("**Detection Details**")
                            if detection_info:
                                for detection in detection_info:
                                    with st.container():
                                        st.markdown(f"""
                                        <div class="detection-highlight">
                                            <strong>Detection {detection['id']}</strong><br>
                                            Center: ({detection['center'][0]}, {detection['center'][1]})<br>
                                            Size: {detection['size'][0]} × {detection['size'][1]}<br>
                                            Importance: {detection['importance']:.3f}
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.write("")
                            else:
                                st.info("No significant detection areas found")
                        
                        st.markdown("""
                        **🟢 Green Circles**: Main detection areas where the CNN focuses
                        **🔴 Red Rectangles**: Bounding boxes of important regions
                        **Numbers**: Detection ID for reference
                        """)
                
                with tab5:
                    st.subheader("📊 Analysis Summary")
                    
                    # Create summary visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Prediction Summary**")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                        
                        # Probability pie chart
                        ax1.pie(prediction, labels=CLASSES, autopct='%1.1f%%', 
                               colors=['#ff6b6b', '#51cf66'])
                        ax1.set_title('Class Probabilities')
                        
                        # Confidence bar
                        ax2.bar(CLASSES, prediction, color=['#ff6b6b', '#51cf66'])
                        ax2.set_title('Prediction Confidence')
                        ax2.set_ylabel('Probability')
                        ax2.set_ylim([0, 1])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.write("**Key Findings**")
                        
                        decision_explanation = f"""
                        **Final Decision: {predicted_class}**
                        
                        🎯 **Confidence Level**: {confidence:.1%}
                        {'🟢 High' if confidence > 0.8 else '🟡 Medium' if confidence > 0.6 else '🔴 Low'}
                        
                        📊 **Fraud Probability**: {prediction[0]:.1%}
                        📊 **Legitimate Probability**: {prediction[1]:.1%}
                        
                        🔍 **Detection Areas**: {len(detection_info) if 'detection_info' in locals() else 0}
                        
                        💡 **Reasoning**: The CNN analyzed multiple layers of features including edges, patterns, and high-level damage characteristics to make this classification.
                        """
                        
                        st.markdown(decision_explanation)
                        
                        # Recommendation
                        if predicted_class == 'Fraud':
                            if confidence > 0.8:
                                st.error("🚨 **HIGH RISK**: Strong indication of fraud. Immediate investigation recommended.")
                            else:
                                st.warning("⚠️ **MEDIUM RISK**: Possible fraud detected. Additional verification needed.")
                        else:
                            if confidence > 0.8:
                                st.success("✅ **LOW RISK**: Appears legitimate. Standard processing approved.")
                            else:
                                st.info("ℹ️ **UNCERTAIN**: Review recommended before processing.")

if __name__ == "__main__":
    main()
