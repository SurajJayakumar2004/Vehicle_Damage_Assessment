"""
Filter Analysis Demo
===================

A demonstration script showing how CNN filters work and what they learn.
This script creates a sample model and shows various visualization techniques.

Run: python filter_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np

def create_demo_filters():
    """Create sample filters that demonstrate common learned patterns"""
    
    # Edge detection filters
    edge_vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    edge_horizontal = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    edge_diagonal = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float32)
    
    # Corner detection
    corner = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32)
    
    # Blur/smoothing
    blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    
    # Sharpening
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    
    # Emboss
    emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    
    # Random learned pattern (simulation)
    random_pattern = np.random.randn(3, 3) * 0.5
    
    return {
        'Vertical Edge': edge_vertical,
        'Horizontal Edge': edge_horizontal,
        'Diagonal Edge': edge_diagonal,
        'Corner Detector': corner,
        'Blur Filter': blur,
        'Sharpen Filter': sharpen,
        'Emboss Filter': emboss,
        'Learned Pattern': random_pattern
    }

def visualize_demo_filters():
    """Visualize the demo filters"""
    filters = create_demo_filters()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Common CNN Filter Patterns (What Your Model Might Learn)', 
                 fontsize=16, fontweight='bold')
    
    filter_names = list(filters.keys())
    
    for i, (name, filter_kernel) in enumerate(filters.items()):
        row = i // 4
        col = i % 4
        
        im = axes[row, col].imshow(filter_kernel, cmap='RdBu', vmin=-2, vmax=2)
        axes[row, col].set_title(name, fontsize=12, fontweight='bold')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        
        # Add values as text
        for y in range(filter_kernel.shape[0]):
            for x in range(filter_kernel.shape[1]):
                value = filter_kernel[y, x]
                color = 'white' if abs(value) > 1.0 else 'black'
                axes[row, col].text(x, y, f'{value:.1f}', 
                                  ha='center', va='center', 
                                  color=color, fontweight='bold')
        
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def create_sample_image():
    """Create a sample image with various features"""
    # Create a 64x64 sample image
    img = np.zeros((64, 64))
    
    # Add vertical lines (edges)
    img[:, 15] = 1.0
    img[:, 45] = 1.0
    
    # Add horizontal lines
    img[20, :] = 1.0
    img[40, :] = 1.0
    
    # Add a square (corners)
    img[25:35, 25:35] = 0.5
    img[25, 25:35] = 1.0  # Top edge
    img[34, 25:35] = 1.0  # Bottom edge
    img[25:35, 25] = 1.0  # Left edge
    img[25:35, 34] = 1.0  # Right edge
    
    # Add some noise
    noise = np.random.random((64, 64)) * 0.1
    img += noise
    
    return img

def apply_filter_to_image(image, filter_kernel):
    """Apply a filter to an image (simplified convolution)"""
    from scipy import ndimage
    
    try:
        # Use scipy for convolution if available
        result = ndimage.convolve(image, filter_kernel, mode='constant')
    except ImportError:
        # Simple implementation without scipy
        result = np.zeros_like(image)
        pad = filter_kernel.shape[0] // 2
        
        for i in range(pad, image.shape[0] - pad):
            for j in range(pad, image.shape[1] - pad):
                # Extract region
                region = image[i-pad:i+pad+1, j-pad:j+pad+1]
                # Apply filter
                result[i, j] = np.sum(region * filter_kernel)
    
    return result

def demonstrate_feature_maps():
    """Demonstrate how filters create feature maps"""
    # Create sample image
    sample_image = create_sample_image()
    
    # Get demo filters
    filters = create_demo_filters()
    
    # Select a few interesting filters
    selected_filters = {
        'Original Image': sample_image,
        'Vertical Edge': apply_filter_to_image(sample_image, filters['Vertical Edge']),
        'Horizontal Edge': apply_filter_to_image(sample_image, filters['Horizontal Edge']),
        'Corner Detector': apply_filter_to_image(sample_image, filters['Corner Detector']),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('How Filters Create Feature Maps', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (name, feature_map) in enumerate(selected_filters.items()):
        im = axes[i].imshow(feature_map, cmap='viridis' if i == 0 else 'RdBu')
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def explain_vehicle_damage_detection():
    """Explain how these concepts apply to vehicle damage detection"""
    
    print("\n🚗 Vehicle Damage Detection - What Your Model Learns")
    print("=" * 60)
    
    explanations = {
        "Early Layers (Edge Detection)": [
            "• Detect scratch lines and dent edges",
            "• Identify paint damage boundaries", 
            "• Find sharp transitions in vehicle surface",
            "• Distinguish between normal panel lines and damage"
        ],
        
        "Middle Layers (Pattern Recognition)": [
            "• Recognize complex damage shapes",
            "• Identify rust patterns and corrosion",
            "• Detect impact damage patterns",
            "• Find inconsistent surface textures"
        ],
        
        "Deep Layers (Fraud Detection)": [
            "• Identify suspicious damage patterns",
            "• Detect artificially created damage",
            "• Recognize inconsistent lighting/shadows",
            "• Find evidence of photo manipulation"
        ]
    }
    
    for category, points in explanations.items():
        print(f"\n{category}:")
        print("-" * 40)
        for point in points:
            print(point)
    
    print(f"\n💡 Key Insights:")
    print("-" * 40)
    print("• Each layer builds on the previous one")
    print("• Early layers find basic features, deep layers find complex patterns")
    print("• Fraud detection relies on inconsistency detection")
    print("• Multiple filters work together to make final decisions")

def show_filter_statistics_demo():
    """Demonstrate filter statistics analysis"""
    filters = create_demo_filters()
    
    print("\n📊 Filter Statistics Analysis")
    print("=" * 60)
    
    for name, filter_kernel in filters.items():
        stats = {
            'Mean': np.mean(filter_kernel),
            'Std Dev': np.std(filter_kernel),
            'Min': np.min(filter_kernel),
            'Max': np.max(filter_kernel),
            'L1 Norm': np.sum(np.abs(filter_kernel)),
            'L2 Norm': np.sqrt(np.sum(filter_kernel**2))
        }
        
        print(f"\n{name}:")
        print("-" * 30)
        for stat_name, value in stats.items():
            print(f"  {stat_name:10}: {value:8.3f}")

def main():
    """Main demonstration function"""
    print("🔍 CNN Filter Analysis Demo")
    print("=" * 40)
    print("This demo shows how CNN filters work and what they learn.")
    print("Perfect for understanding your Vehicle Damage Assessment model!\n")
    
    try:
        # Show filter visualizations
        print("1. 🎨 Visualizing Common Filter Patterns...")
        visualize_demo_filters()
        
        # Show feature maps
        print("\n2. 🖼️ Demonstrating Feature Map Creation...")
        demonstrate_feature_maps()
        
        # Show statistics
        print("\n3. 📊 Filter Statistics Analysis...")
        show_filter_statistics_demo()
        
        # Explain vehicle damage context
        print("\n4. 🚗 Vehicle Damage Detection Context...")
        explain_vehicle_damage_detection()
        
        print("\n✅ Demo Complete!")
        print("Now you understand what your CNN model learns!")
        print("Use the other tools to analyze your actual trained model.")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        print("Make sure matplotlib is installed: pip install matplotlib")

if __name__ == "__main__":
    main()