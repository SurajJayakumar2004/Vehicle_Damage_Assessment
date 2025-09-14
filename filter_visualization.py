"""
CNN Filter Visualization Tool
============================

This tool helps visualize what each convolutional filter in your Vehicle Damage Assessment
fraud detection model has learned. It shows feature maps, filter kernels, and activation
patterns to understand how the model processes vehicle damage images.

Author: Vehicle Damage Assessment AI
Date: September 2025
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import pandas as pd

# Set page config
st.set_page_config(
    page_title="CNN Filter Visualization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .filter-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
    
    .layer-info {
        background: rgba(74, 144, 226, 0.1);
        border-left: 4px solid #4A90E2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 CNN Filter Visualization Tool</h1>
    <p>Explore what your Vehicle Damage Assessment model has learned</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try to load the model from the common locations
        model_paths = [
            "vehicle_damage_fraud_model.h5",
            "model/vehicle_damage_fraud_model.h5",
            "models/vehicle_damage_fraud_model.h5"
        ]
        
        for path in model_paths:
            try:
                model = tf.keras.models.load_model(path)
                st.success(f"✅ Model loaded successfully from: {path}")
                return model
            except:
                continue
                
        # If no model found, create a sample model architecture
        st.warning("⚠️ Model file not found. Creating sample architecture for demonstration.")
        return create_sample_model()
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return create_sample_model()

def create_sample_model():
    """Create a sample model for demonstration"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), name='conv2d_1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_1'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_2'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_3'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv2d_4'),
        tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    return model

def get_layer_info(model):
    """Extract information about convolutional layers"""
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if 'conv' in layer.name.lower():
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': type(layer).__name__,
                'filters': layer.filters,
                'kernel_size': layer.kernel_size,
                'activation': layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation),
                'output_shape': layer.output_shape
            }
            conv_layers.append(layer_info)
    return conv_layers

def visualize_filters(model, layer_name, num_filters=16):
    """Visualize the learned filters from a specific layer"""
    try:
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()[0]  # Get the filter weights
        
        # Normalize weights for visualization
        weights = (weights - weights.mean()) / weights.std()
        weights = np.clip(weights, -2, 2)
        weights = (weights + 2) / 4  # Normalize to 0-1
        
        num_filters = min(num_filters, weights.shape[-1])
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=4,
            subplot_titles=[f'Filter {i+1}' for i in range(num_filters)],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        for i in range(num_filters):
            row = i // 4 + 1
            col = i % 4 + 1
            
            # Get the filter (taking the first channel if RGB)
            filter_img = weights[:, :, 0, i] if weights.shape[2] > 1 else weights[:, :, i]
            
            fig.add_trace(
                go.Heatmap(
                    z=filter_img,
                    colorscale='RdBu',
                    showscale=False,
                    hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f"Learned Filters in {layer_name}",
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig
        
    except Exception as e:
        st.error(f"Error visualizing filters: {str(e)}")
        return None

def create_feature_maps(model, image, layer_name):
    """Create feature maps for a given image and layer"""
    try:
        # Create a model that outputs the feature maps
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get feature maps
        feature_maps = feature_model(image)
        
        return feature_maps.numpy()
        
    except Exception as e:
        st.error(f"Error creating feature maps: {str(e)}")
        return None

def visualize_feature_maps(feature_maps, layer_name, num_maps=16):
    """Visualize feature maps"""
    if feature_maps is None:
        return None
        
    try:
        # Take the first sample if batch dimension exists
        if len(feature_maps.shape) == 4:
            feature_maps = feature_maps[0]
            
        num_maps = min(num_maps, feature_maps.shape[-1])
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=4,
            subplot_titles=[f'Feature Map {i+1}' for i in range(num_maps)],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        for i in range(num_maps):
            row = i // 4 + 1
            col = i % 4 + 1
            
            feature_map = feature_maps[:, :, i]
            
            fig.add_trace(
                go.Heatmap(
                    z=feature_map,
                    colorscale='Viridis',
                    showscale=False,
                    hovertemplate='x: %{x}<br>y: %{y}<br>activation: %{z:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f"Feature Maps from {layer_name}",
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig
        
    except Exception as e:
        st.error(f"Error visualizing feature maps: {str(e)}")
        return None

def analyze_filter_statistics(model, layer_name):
    """Analyze statistical properties of filters"""
    try:
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()[0]
        
        stats = {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights),
            'sparsity': np.mean(np.abs(weights) < 0.01),  # Percentage of near-zero weights
            'l1_norm': np.mean(np.abs(weights)),
            'l2_norm': np.sqrt(np.mean(weights**2))
        }
        
        return stats
        
    except Exception as e:
        st.error(f"Error analyzing filter statistics: {str(e)}")
        return None

# Main app
def main():
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Could not load model. Please check your model file.")
        return
    
    # Get layer information
    conv_layers = get_layer_info(model)
    
    if not conv_layers:
        st.error("❌ No convolutional layers found in the model.")
        return
    
    # Layer selection
    layer_names = [layer['name'] for layer in conv_layers]
    selected_layer = st.sidebar.selectbox(
        "🔍 Select Layer to Analyze",
        layer_names,
        help="Choose a convolutional layer to visualize its filters"
    )
    
    # Number of filters to show
    num_filters = st.sidebar.slider(
        "📊 Number of Filters to Display",
        min_value=4,
        max_value=64,
        value=16,
        step=4,
        help="Select how many filters to visualize"
    )
    
    # Analysis type
    analysis_type = st.sidebar.radio(
        "📈 Analysis Type",
        ["Filter Kernels", "Feature Maps", "Filter Statistics", "Layer Comparison"],
        help="Choose the type of analysis to perform"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🏗️ Model Architecture")
        
        # Display layer information
        for layer in conv_layers:
            st.markdown(f"""
            <div class="layer-info">
                <h4>{layer['name']}</h4>
                <p><strong>Filters:</strong> {layer['filters']}</p>
                <p><strong>Kernel Size:</strong> {layer['kernel_size']}</p>
                <p><strong>Activation:</strong> {layer['activation']}</p>
                <p><strong>Output Shape:</strong> {layer['output_shape']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col1:
        if analysis_type == "Filter Kernels":
            st.markdown("### 🔍 Learned Filter Kernels")
            st.markdown(f"Visualizing filters from **{selected_layer}**")
            
            # Visualize filters
            filter_fig = visualize_filters(model, selected_layer, num_filters)
            if filter_fig:
                st.plotly_chart(filter_fig, use_container_width=True)
                
                st.markdown("""
                **What you're seeing:**
                - Each heatmap shows the weights of a learned filter
                - Darker colors indicate negative weights, lighter colors indicate positive weights
                - Patterns reveal what features the filter is designed to detect
                - Early layers typically detect edges, corners, and simple textures
                - Deeper layers detect more complex patterns and shapes
                """)
        
        elif analysis_type == "Feature Maps":
            st.markdown("### 🖼️ Feature Map Visualization")
            
            # Image upload for feature map generation
            uploaded_image = st.file_uploader(
                "Upload an image to see feature maps",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a vehicle damage image to see how the model processes it"
            )
            
            if uploaded_image is not None:
                # Process image
                image = Image.open(uploaded_image)
                st.image(image, caption="Input Image", width=300)
                
                # Preprocess image for model
                img_array = np.array(image.resize((224, 224))) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Generate feature maps
                feature_maps = create_feature_maps(model, img_array, selected_layer)
                feature_fig = visualize_feature_maps(feature_maps, selected_layer, num_filters)
                
                if feature_fig:
                    st.plotly_chart(feature_fig, use_container_width=True)
                    
                    st.markdown("""
                    **What you're seeing:**
                    - Bright areas show where the filter activates strongly
                    - Dark areas show little to no activation
                    - Different feature maps highlight different aspects of the image
                    - Early layers show edge detection, later layers show complex patterns
                    """)
        
        elif analysis_type == "Filter Statistics":
            st.markdown("### 📊 Filter Statistical Analysis")
            
            stats = analyze_filter_statistics(model, selected_layer)
            
            if stats:
                # Display metrics
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{stats['mean']:.4f}</h3>
                        <p>Mean Weight</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[1]:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{stats['std']:.4f}</h3>
                        <p>Standard Deviation</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[2]:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>{stats['sparsity']:.2%}</h3>
                        <p>Sparsity</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional statistics
                st.markdown("#### Detailed Statistics")
                stats_df = pd.DataFrame([
                    {"Metric": "Minimum Weight", "Value": f"{stats['min']:.6f}"},
                    {"Metric": "Maximum Weight", "Value": f"{stats['max']:.6f}"},
                    {"Metric": "L1 Norm", "Value": f"{stats['l1_norm']:.6f}"},
                    {"Metric": "L2 Norm", "Value": f"{stats['l2_norm']:.6f}"},
                ])
                st.dataframe(stats_df, use_container_width=True)
                
                st.markdown("""
                **Statistics Explanation:**
                - **Mean Weight**: Average value of all filter weights
                - **Standard Deviation**: Measure of weight variation
                - **Sparsity**: Percentage of near-zero weights (indicates pruning potential)
                - **L1/L2 Norms**: Measures of filter magnitude (useful for regularization analysis)
                """)
        
        elif analysis_type == "Layer Comparison":
            st.markdown("### ⚖️ Layer Comparison")
            
            # Compare statistics across layers
            layer_stats = []
            for layer_info in conv_layers:
                stats = analyze_filter_statistics(model, layer_info['name'])
                if stats:
                    stats['layer'] = layer_info['name']
                    stats['filters'] = layer_info['filters']
                    layer_stats.append(stats)
            
            if layer_stats:
                df = pd.DataFrame(layer_stats)
                
                # Plot comparisons
                fig_stats = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Mean Weights', 'Standard Deviation', 'Sparsity', 'Number of Filters']
                )
                
                # Mean weights
                fig_stats.add_trace(
                    go.Bar(x=df['layer'], y=df['mean'], name='Mean'),
                    row=1, col=1
                )
                
                # Standard deviation
                fig_stats.add_trace(
                    go.Bar(x=df['layer'], y=df['std'], name='Std Dev'),
                    row=1, col=2
                )
                
                # Sparsity
                fig_stats.add_trace(
                    go.Bar(x=df['layer'], y=df['sparsity'], name='Sparsity'),
                    row=2, col=1
                )
                
                # Number of filters
                fig_stats.add_trace(
                    go.Bar(x=df['layer'], y=df['filters'], name='Filters'),
                    row=2, col=2
                )
                
                fig_stats.update_layout(height=600, showlegend=False, title="Layer Statistics Comparison")
                fig_stats.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig_stats, use_container_width=True)
                
                st.markdown("""
                **Layer Comparison Insights:**
                - Earlier layers typically have smaller, more focused filters
                - Later layers often show higher sparsity (more specialized)
                - Standard deviation can indicate filter diversity within each layer
                - Filter count usually increases with depth for feature hierarchy
                """)

# Run the app
if __name__ == "__main__":
    main()