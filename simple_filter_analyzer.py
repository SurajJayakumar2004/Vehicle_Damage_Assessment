"""
Simple CNN Filter Analyzer
==========================

A simplified tool to analyze and visualize CNN filters without complex dependencies.
This version focuses on core functionality and can work with basic libraries.

Usage:
    python simple_filter_analyzer.py --model_path your_model.h5 --layer conv2d_1
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import argparse
import os

class FilterVisualizer:
    """Simple CNN Filter Visualization Tool"""
    
    def __init__(self, model_path=None):
        """Initialize the visualizer with a model"""
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"✅ Model loaded successfully from: {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {str(e)}")
                self.model = self._create_sample_model()
        else:
            print("⚠️ Creating sample model for demonstration")
            self.model = self._create_sample_model()
    
    def _create_sample_model(self):
        """Create a sample CNN model for demonstration"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                 input_shape=(224, 224, 3), name='conv2d_1'),
            tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_1'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
            tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_2'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid', name='output')
        ])
        return model
    
    def get_conv_layers(self):
        """Get all convolutional layers from the model"""
        conv_layers = []
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append({
                    'index': i,
                    'name': layer.name,
                    'filters': layer.filters,
                    'kernel_size': layer.kernel_size,
                    'activation': layer.activation.__name__,
                    'output_shape': layer.output_shape
                })
        return conv_layers
    
    def visualize_filters(self, layer_name, num_filters=16, save_path=None):
        """Visualize the learned filters from a specific layer"""
        try:
            layer = self.model.get_layer(layer_name)
            weights, biases = layer.get_weights()
            
            # Normalize weights for better visualization
            weights_norm = (weights - weights.mean()) / (weights.std() + 1e-8)
            
            # Determine subplot layout
            num_filters = min(num_filters, weights.shape[-1])
            cols = 4
            rows = (num_filters + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
            fig.suptitle(f'Learned Filters in {layer_name}', fontsize=16, fontweight='bold')
            
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_filters):
                row = i // cols
                col = i % cols
                
                # Get filter weights (use first input channel if multiple)
                if weights.shape[2] > 1:
                    filter_weights = weights_norm[:, :, 0, i]
                else:
                    filter_weights = weights_norm[:, :, 0, i]
                
                # Plot filter
                im = axes[row, col].imshow(filter_weights, cmap='RdBu', aspect='auto')
                axes[row, col].set_title(f'Filter {i+1}', fontsize=10)
                axes[row, col].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            # Hide empty subplots
            for i in range(num_filters, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Filter visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error visualizing filters: {str(e)}")
    
    def create_feature_maps(self, image_path, layer_name, save_path=None):
        """Create and visualize feature maps for a given image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image = image.resize((224, 224))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Create intermediate model
            intermediate_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            
            # Get feature maps
            feature_maps = intermediate_model.predict(img_array)
            feature_maps = feature_maps[0]  # Remove batch dimension
            
            # Visualize feature maps
            num_maps = min(16, feature_maps.shape[-1])
            cols = 4
            rows = (num_maps + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols + 1, figsize=(20, rows * 4))
            fig.suptitle(f'Feature Maps from {layer_name}', fontsize=16, fontweight='bold')
            
            # Show original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
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
                
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            # Hide empty subplots
            for i in range(num_maps, rows * cols):
                row = i // cols
                col = i % cols + 1
                if col < axes.shape[1]:
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Feature maps saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating feature maps: {str(e)}")
    
    def analyze_layer_statistics(self, layer_name):
        """Analyze statistical properties of a layer's filters"""
        try:
            layer = self.model.get_layer(layer_name)
            weights = layer.get_weights()[0]
            
            stats = {
                'Layer Name': layer_name,
                'Shape': weights.shape,
                'Total Parameters': np.prod(weights.shape),
                'Mean': np.mean(weights),
                'Std Dev': np.std(weights),
                'Min': np.min(weights),
                'Max': np.max(weights),
                'Sparsity (%)': np.mean(np.abs(weights) < 0.01) * 100,
                'L1 Norm': np.sum(np.abs(weights)),
                'L2 Norm': np.sqrt(np.sum(weights**2))
            }
            
            print(f"\n📊 Statistics for {layer_name}:")
            print("-" * 40)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key:20}: {value:.6f}")
                else:
                    print(f"{key:20}: {value}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error analyzing layer statistics: {str(e)}")
            return None
    
    def compare_all_layers(self):
        """Compare statistics across all convolutional layers"""
        conv_layers = self.get_conv_layers()
        
        print("\n🔍 Model Architecture Overview:")
        print("=" * 60)
        
        for layer_info in conv_layers:
            print(f"\nLayer: {layer_info['name']}")
            print(f"  Filters: {layer_info['filters']}")
            print(f"  Kernel Size: {layer_info['kernel_size']}")
            print(f"  Activation: {layer_info['activation']}")
            print(f"  Output Shape: {layer_info['output_shape']}")
            
            # Get statistics
            self.analyze_layer_statistics(layer_info['name'])
    
    def generate_report(self, output_dir="filter_analysis_report"):
        """Generate a comprehensive analysis report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        conv_layers = self.get_conv_layers()
        
        print(f"\n📋 Generating comprehensive report in: {output_dir}")
        print("=" * 60)
        
        # Create filter visualizations for each layer
        for layer_info in conv_layers:
            layer_name = layer_info['name']
            print(f"Analyzing layer: {layer_name}")
            
            # Visualize filters
            filter_path = os.path.join(output_dir, f"{layer_name}_filters.png")
            self.visualize_filters(layer_name, save_path=filter_path)
            
            # Analyze statistics
            self.analyze_layer_statistics(layer_name)
        
        print(f"\n✅ Analysis complete! Check the '{output_dir}' folder for results.")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='CNN Filter Visualization Tool')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--layer', type=str, help='Specific layer to analyze')
    parser.add_argument('--image_path', type=str, help='Path to image for feature map generation')
    parser.add_argument('--num_filters', type=int, default=16, help='Number of filters to visualize')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = FilterVisualizer(args.model_path)
    
    if args.report:
        # Generate comprehensive report
        visualizer.generate_report()
    elif args.layer:
        # Analyze specific layer
        print(f"\n🔍 Analyzing layer: {args.layer}")
        
        # Visualize filters
        visualizer.visualize_filters(args.layer, args.num_filters)
        
        # Show statistics
        visualizer.analyze_layer_statistics(args.layer)
        
        # Generate feature maps if image provided
        if args.image_path:
            visualizer.create_feature_maps(args.image_path, args.layer)
    else:
        # Show model overview
        visualizer.compare_all_layers()

if __name__ == "__main__":
    # Interactive mode if no command line arguments
    import sys
    if len(sys.argv) == 1:
        print("🔍 CNN Filter Visualization Tool")
        print("=" * 40)
        
        visualizer = FilterVisualizer()
        
        # Show available layers
        conv_layers = visualizer.get_conv_layers()
        print(f"\nFound {len(conv_layers)} convolutional layers:")
        for i, layer in enumerate(conv_layers):
            print(f"  {i+1}. {layer['name']} ({layer['filters']} filters)")
        
        # Interactive selection
        try:
            choice = input(f"\nSelect layer to analyze (1-{len(conv_layers)}) or 'all' for report: ")
            
            if choice.lower() == 'all':
                visualizer.generate_report()
            else:
                layer_idx = int(choice) - 1
                if 0 <= layer_idx < len(conv_layers):
                    layer_name = conv_layers[layer_idx]['name']
                    visualizer.visualize_filters(layer_name)
                    visualizer.analyze_layer_statistics(layer_name)
                else:
                    print("❌ Invalid selection")
                    
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
    else:
        main()