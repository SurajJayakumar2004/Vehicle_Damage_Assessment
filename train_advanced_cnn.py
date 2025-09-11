#!/usr/bin/env python3
"""
Advanced CNN Training with Visualization and Feature Analysis
Shows layer outputs, edge detection, and feature maps for fraud detection
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=== Advanced Vehicle Damage CNN with Visualization ===")
print(f"TensorFlow version: {tf.__version__}")

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
CLASSES = ['Fraud', 'Non-Fraud']
NUM_CLASSES = len(CLASSES)

# Data paths
TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
MODEL_SAVE_PATH = 'models/advanced_cnn_model.h5'
VISUALIZATION_DIR = 'visualizations'

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_and_preprocess_data(data_path, img_size=IMG_SIZE, max_per_class=None):
    """Load images and labels with optional limit per class"""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found!")
            continue
            
        print(f"Loading {class_name} images from {class_path}...")
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_per_class:
            image_files = image_files[:max_per_class]
        
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

def create_advanced_cnn_model(input_shape, num_classes):
    """Create an advanced CNN with named layers for visualization"""
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape, name='input_layer'),
        
        # First convolutional block - Edge Detection
        layers.Conv2D(32, (3, 3), activation='relu', name='conv1_edge_detection'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block - Pattern Recognition
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2_patterns'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third convolutional block - Feature Combination
        layers.Conv2D(128, (3, 3), activation='relu', name='conv3_features'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Fourth convolutional block - High-level Features
        layers.Conv2D(256, (3, 3), activation='relu', name='conv4_highlevel'),
        layers.BatchNormalization(name='bn4'),
        layers.MaxPooling2D((2, 2), name='pool4'),
        
        # Flatten and classification layers
        layers.Flatten(name='flatten'),
        layers.Dropout(0.5, name='dropout1'),
        layers.Dense(512, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn_dense'),
        layers.Dropout(0.3, name='dropout2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

def visualize_layer_outputs(model, image, layer_names, save_path):
    """Visualize outputs of specified layers"""
    
    # Create models for each layer
    layer_outputs = []
    for layer_name in layer_names:
        try:
            intermediate_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(layer_name).output)
            output = intermediate_model.predict(np.expand_dims(image, axis=0), verbose=0)
            layer_outputs.append((layer_name, output))
        except:
            print(f"Could not get output for layer: {layer_name}")
            continue
    
    # Create visualization
    fig, axes = plt.subplots(len(layer_outputs), 8, figsize=(20, len(layer_outputs) * 3))
    
    for layer_idx, (layer_name, output) in enumerate(layer_outputs):
        # Show first 8 feature maps
        for i in range(min(8, output.shape[-1])):
            ax = axes[layer_idx, i] if len(layer_outputs) > 1 else axes[i]
            
            feature_map = output[0, :, :, i]
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'{layer_name}\nFilter {i+1}', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return layer_outputs

def analyze_edge_detection(image, model):
    """Analyze edge detection in the first convolutional layer"""
    
    # Get first conv layer output
    conv1_model = keras.Model(inputs=model.input,
                             outputs=model.get_layer('conv1_edge_detection').output)
    conv1_output = conv1_model.predict(np.expand_dims(image, axis=0), verbose=0)
    
    # Also apply traditional edge detection for comparison
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges_canny = cv2.Canny(gray_image, 50, 150)
    edges_sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    
    # Create comparison visualization
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Traditional edge detection
    axes[0, 1].imshow(edges_canny, cmap='gray')
    axes[0, 1].set_title('Canny Edge Detection')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(edges_sobel, cmap='gray')
    axes[0, 2].set_title('Sobel Edge Detection')
    axes[0, 2].axis('off')
    
    # CNN learned filters (first few)
    for i in range(3, 6):
        if i-3 < conv1_output.shape[-1]:
            axes[0, i].imshow(conv1_output[0, :, :, i-3], cmap='viridis')
            axes[0, i].set_title(f'CNN Filter {i-2}')
            axes[0, i].axis('off')
    
    # Show more CNN filters
    for i in range(12):
        row = 1 + i // 6
        col = i % 6
        if i < conv1_output.shape[-1] and row < 3:
            axes[row, col].imshow(conv1_output[0, :, :, i], cmap='viridis')
            axes[row, col].set_title(f'CNN Filter {i+1}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig, conv1_output

def create_attention_map(model, image, class_idx):
    """Create attention map showing what parts of image the model focuses on"""
    
    # Get the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
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
        class_prob = predictions[:, class_idx]
    
    # Calculate gradients of class probability with respect to conv layer output
    grads = tape.gradient(class_prob, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by gradients
    conv_outputs = conv_outputs[0]
    for i in range(pooled_grads.shape[-1]):
        conv_outputs = conv_outputs[:, :, i:i+1] * pooled_grads[i]
    
    # Create heatmap
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def highlight_detection_areas(image, heatmap, threshold=0.5):
    """Highlight important areas on the original image"""
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Create colored overlay
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = cv2.addWeighted(
        (image * 255).astype(np.uint8), 
        0.7, 
        heatmap_colored, 
        0.3, 
        0
    )
    
    # Find contours for high-attention areas
    binary_mask = (heatmap_resized > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw circles/rectangles around detected areas
    result_image = overlay.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small areas
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw circle around center
            center_x, center_y = x + w//2, y + h//2
            radius = max(w, h) // 2
            cv2.circle(result_image, (center_x, center_y), radius, (0, 255, 0), 2)
    
    return result_image, overlay, heatmap_colored

def comprehensive_evaluation(model, X_test, y_test):
    """Comprehensive model evaluation with multiple metrics"""
    
    print("\n=== Comprehensive Model Evaluation ===")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_proba = y_pred[:, 1]  # Probability of positive class (Non-Fraud)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{VISUALIZATION_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    y_test_binary = label_binarize(y_test, classes=[0, 1])
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fraud Detection Performance')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{VISUALIZATION_DIR}/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate additional metrics
    TP = cm[1, 1]  # True Positives
    TN = cm[0, 0]  # True Negatives
    FP = cm[0, 1]  # False Positives
    FN = cm[1, 0]  # False Negatives
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }
    
    print(f"\nDetailed Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    
    return metrics

def analyze_sample_predictions(model, X_test, y_test, num_samples=6):
    """Analyze predictions on sample test images with full visualization"""
    
    print(f"\n=== Analyzing {num_samples} Sample Predictions ===")
    
    # Select diverse samples
    fraud_indices = np.where(y_test == 0)[0][:num_samples//2]
    non_fraud_indices = np.where(y_test == 1)[0][:num_samples//2]
    sample_indices = np.concatenate([fraud_indices, non_fraud_indices])
    
    for i, idx in enumerate(sample_indices):
        image = X_test[idx]
        true_label = y_test[idx]
        true_class = CLASSES[true_label]
        
        # Get prediction
        prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        predicted_class_idx = np.argmax(prediction)
        predicted_class = CLASSES[predicted_class_idx]
        confidence = prediction[predicted_class_idx]
        
        print(f"\nSample {i+1}: True={true_class}, Predicted={predicted_class}, Confidence={confidence:.3f}")
        
        # 1. Layer-by-layer analysis
        layer_names = ['conv1_edge_detection', 'conv2_patterns', 'conv3_features', 'conv4_highlevel']
        layer_outputs = visualize_layer_outputs(
            model, image, layer_names, 
            f'{VISUALIZATION_DIR}/sample_{i+1}_layer_outputs.png'
        )
        
        # 2. Edge detection analysis
        edge_fig, conv1_output = analyze_edge_detection(image, model)
        edge_fig.savefig(f'{VISUALIZATION_DIR}/sample_{i+1}_edge_analysis.png', 
                        dpi=150, bbox_inches='tight')
        plt.close(edge_fig)
        
        # 3. Attention map and highlighting
        heatmap = create_attention_map(model, image, predicted_class_idx)
        if heatmap is not None:
            highlighted_image, overlay, heatmap_colored = highlight_detection_areas(image, heatmap)
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Original, Heatmap, Overlay
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(heatmap, cmap='hot')
            axes[0, 1].set_title('Attention Heatmap')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(overlay / 255.0)
            axes[0, 2].set_title('Heatmap Overlay')
            axes[0, 2].axis('off')
            
            # Row 2: Highlighted areas, prediction info
            axes[1, 0].imshow(highlighted_image / 255.0)
            axes[1, 0].set_title('Detection Areas (Circled)')
            axes[1, 0].axis('off')
            
            # Prediction details
            axes[1, 1].axis('off')
            prediction_text = f"""
Prediction Analysis:

True Class: {true_class}
Predicted: {predicted_class}
Confidence: {confidence:.3f}

Fraud Prob: {prediction[0]:.3f}
Non-Fraud Prob: {prediction[1]:.3f}

Status: {'✅ Correct' if true_class == predicted_class else '❌ Incorrect'}
            """
            axes[1, 1].text(0.1, 0.5, prediction_text, fontsize=12, 
                           verticalalignment='center', transform=axes[1, 1].transAxes)
            
            # Feature importance
            axes[1, 2].bar(['Fraud', 'Non-Fraud'], prediction, color=['red', 'green'], alpha=0.7)
            axes[1, 2].set_title('Class Probabilities')
            axes[1, 2].set_ylabel('Probability')
            axes[1, 2].set_ylim([0, 1])
            
            plt.suptitle(f'Sample {i+1} - Complete Analysis', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{VISUALIZATION_DIR}/sample_{i+1}_complete_analysis.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()

def main():
    """Main training and analysis function"""
    
    print("Loading training data...")
    X_train, y_train = load_and_preprocess_data(TRAIN_PATH)
    print(f"Training data: {X_train.shape}")
    
    print("Loading test data...")
    X_test, y_test = load_and_preprocess_data(TEST_PATH)
    print(f"Test data: {X_test.shape}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ No data loaded! Check your data directories.")
        return
    
    # Convert labels to categorical
    y_train_categorical = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_categorical = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    # Create advanced model
    print("\n=== Creating Advanced CNN Model ===")
    model = create_advanced_cnn_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    
    # Compile with advanced optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("Model Architecture:")
    model.summary()
    
    # Training with callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]
    
    print(f"\n=== Training Advanced CNN Model ===")
    history = model.fit(
        X_train, y_train_categorical,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_categorical),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_DIR}/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Comprehensive evaluation
    metrics = comprehensive_evaluation(model, X_test, y_test)
    
    # Sample analysis with full visualization
    analyze_sample_predictions(model, X_test, y_test, num_samples=6)
    
    # Save model and metadata
    model.save(MODEL_SAVE_PATH)
    
    model_info = {
        'model_path': MODEL_SAVE_PATH,
        'input_shape': (IMG_SIZE, IMG_SIZE, 3),
        'classes': CLASSES,
        'num_classes': NUM_CLASSES,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open('models/advanced_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print(f"Test accuracy: {metrics['accuracy']:.1%}")
    print(f"Visualizations saved in: {VISUALIZATION_DIR}/")
    print(f"Check the visualization files to see:")
    print(f"  - Layer-by-layer feature maps")
    print(f"  - Edge detection analysis")
    print(f"  - Attention maps showing what the model focuses on")
    print(f"  - Highlighted detection areas with circles")

if __name__ == "__main__":
    main()
