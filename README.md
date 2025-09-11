# Vehicle Damage Assessment - CNN Fraud Detection System

A modern deep learning solution for detecting fraudulent vehicle damage claims using Convolutional Neural Networks (CNN) with TensorFlow 2.x.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/Vehicle_Damage_Assessment.git
cd Vehicle_Damage_Assessment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Run the training notebook
jupyter notebook notebooks/vehicle_damage_cnn.ipynb
```

### 3. Run the Web Application
```bash
# Start the Streamlit app
cd src
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
Vehicle_Damage_Assessment/
├── 📁 data/                    # Training and testing datasets
│   ├── train/
│   │   ├── Fraud/             # Fraudulent damage images
│   │   └── Non-Fraud/         # Legitimate damage images
│   └── test/
│       ├── Fraud/
│       └── Non-Fraud/
├── 📁 models/                  # Trained model files
│   ├── vehicle_damage_model.keras      # Main model (Keras format)
│   └── vehicle_damage_cnn_model.h5     # Backup model (HDF5 format)
├── 📁 notebooks/               # Jupyter notebooks
│   └── vehicle_damage_cnn.ipynb        # Main training notebook
├── 📁 src/                     # Source code
│   ├── streamlit_app.py       # Web application
│   └── dataset.py             # Data loading utilities
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎯 Features

### Core Functionality
- **CNN-Based Classification**: Deep learning model for fraud detection
- **Evidence-Based Analysis**: Step-by-step reasoning for predictions
- **Real-time Processing**: Upload and analyze images instantly
- **High Accuracy**: 96%+ accuracy on test dataset

### Web Interface
- **Modern UI**: Clean, intuitive Streamlit interface
- **Image Upload**: Drag-and-drop or click to upload
- **Detailed Analysis**: Layer-by-layer CNN interpretation
- **Risk Assessment**: Color-coded fraud probability indicators

## 🔬 Technical Details

### Model Architecture
- **3 Convolutional Layers**: Progressive feature extraction
- **MaxPooling**: Spatial dimension reduction
- **Dropout Regularization**: Prevents overfitting
- **Dense Layers**: Final classification
- **Softmax Output**: Probability distribution

### Training Specs
- **Framework**: TensorFlow 2.x / Keras
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Iterations**: 7,500 training iterations

## 🎮 Usage Examples

### Training the Model
```python
# In the notebook
train(num_iterations=7500)
print_validation_accuracy(show_confusion_matrix=True)
```

### Using the Web App
1. Open `http://localhost:8501` in your browser
2. Upload a vehicle damage image
3. View fraud probability and evidence analysis
4. Interpret the CNN layer-by-layer reasoning

## 📊 Performance Metrics

- **Overall Accuracy**: 96.0%
- **Fraud Detection Precision**: 95.9%
- **Fraud Detection Recall**: 96.0%
- **AUC Score**: 95.7%

## 🛠️ Development

### Environment Setup
```bash
# Development dependencies
pip install jupyter matplotlib seaborn
```

### File Organization
- Keep models in `/models` directory
- Source code in `/src` directory  
- Notebooks in `/notebooks` directory
- Training data in `/data` directory

## 🔧 Configuration

### Model Parameters
- Image size: 128x128 pixels
- Batch size: 32
- Validation split: 16%
- Input channels: 3 (RGB)

### Environment Variables
No special environment variables required.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the notebook documentation
- Review the training logs

---

**Built with ❤️ using TensorFlow, Streamlit, and modern deep learning techniques.**

---

## 🧠 **How the Neural Network Works**

### **1. Input Processing**
- **Image Format**: 128×128×3 RGB vehicle damage images
- **Data Preprocessing**: Images normalized to [0,1] range using `/255` division
- **Class Labels**: Binary classification - `['Fraud', 'Non-Fraud']`
- **Batch Processing**: Images processed in batches of 32 for efficient training

### **2. CNN Architecture Deep Dive**

#### **Layer 1: Convolutional Layer 1**
```
Input: 128×128×3 (RGB Image)
Filter Size: 3×3
Number of Filters: 32
Activation: ReLU
Pooling: 2×2 Max Pooling
Output: 64×64×32
```
- **Purpose**: Detects basic features like edges, corners, and simple patterns
- **Feature Maps**: Creates 32 different feature maps, each highlighting different aspects
- **Receptive Field**: 3×3 sliding window captures local patterns

#### **Layer 2: Convolutional Layer 2**
```
Input: 64×64×32
Filter Size: 3×3
Number of Filters: 32
Activation: ReLU
Pooling: 2×2 Max Pooling
Output: 32×32×32
```
- **Purpose**: Combines basic features to detect more complex patterns
- **Feature Learning**: Learns combinations of edges to form shapes and textures
- **Spatial Reduction**: Further reduces spatial dimensions while preserving important features

#### **Layer 3: Convolutional Layer 3**
```
Input: 32×32×32
Filter Size: 3×3
Number of Filters: 64
Activation: ReLU
Pooling: 2×2 Max Pooling
Output: 16×16×64
```
- **Purpose**: Captures high-level features specific to damage patterns
- **Deep Features**: Learns complex combinations that might indicate fraud vs legitimate damage
- **Increased Filters**: 64 filters capture more diverse and complex patterns

#### **Layer 4: Flatten Layer**
```
Input: 16×16×64 = 16,384 features
Output: 1×16,384 (flattened vector)
```
- **Purpose**: Converts 3D feature maps to 1D vector for fully connected layers
- **Information Preservation**: Maintains all learned features in linear format

#### **Layer 5: Fully Connected Layer**
```
Input: 16,384 features
Neurons: 128
Activation: ReLU
Dropout: Applied during training to prevent overfitting
Output: 128 features
```
- **Purpose**: Learns complex non-linear combinations of all extracted features
- **Decision Making**: Combines all visual patterns to make classification decisions
- **Feature Integration**: Synthesizes low-level and high-level features

#### **Layer 6: Output Layer**
```
Input: 128 features
Neurons: 2 (Fraud, Non-Fraud)
Activation: Softmax
Output: [probability_fraud, probability_non_fraud]
```
- **Purpose**: Final classification decision with probability scores
- **Softmax**: Ensures probabilities sum to 1.0
- **Binary Classification**: Outputs confidence for each class

---

## 🔍 **How Classification Works**

### **Feature Learning Process**
1. **Edge Detection**: Layer 1 filters detect basic edges and textures in damage images
2. **Pattern Recognition**: Layer 2 combines edges to recognize shapes and damage patterns
3. **Complex Feature Extraction**: Layer 3 identifies sophisticated damage characteristics
4. **Decision Integration**: Fully connected layers combine all features for final classification

### **Fraud vs Legitimate Classification**
The network learns to distinguish between:

**Fraudulent Damage Indicators**:
- Inconsistent damage patterns
- Artificial or staged damage appearances
- Unusual damage locations or severity
- Patterns that don't match typical accident scenarios

**Legitimate Damage Indicators**:
- Natural impact patterns
- Consistent damage spread
- Realistic deformation and paint scratches
- Damage consistent with reported accident scenarios

### **Prediction Process**
1. **Image Input**: New vehicle damage image (128×128×3)
2. **Feature Extraction**: CNN layers extract hierarchical features
3. **Classification**: Final layer outputs probabilities [P_fraud, P_legitimate]
4. **Decision**: Higher probability determines classification
5. **Confidence Score**: Probability value indicates model confidence

---

## 📊 **Training Process**

### **Dataset Configuration**
- **Training Set**: ~84% of data for learning patterns
- **Validation Set**: 16% for monitoring performance during training
- **Test Set**: Separate set for final evaluation
- **Batch Size**: 32 images processed simultaneously

### **Optimization Details**
- **Loss Function**: Cross-entropy loss for binary classification
- **Optimizer**: Adam optimizer with adaptive learning rates
- **Performance Metrics**: Accuracy, validation loss tracking
- **Early Stopping**: Prevents overfitting by monitoring validation performance

### **Training Monitoring**
```python
# Training output example:
Epoch 1 --- Training Accuracy: 75.2%, Validation Accuracy: 73.1%, Validation Loss: 0.523
Epoch 2 --- Training Accuracy: 82.4%, Validation Accuracy: 79.8%, Validation Loss: 0.445
...
```

---

## 🛠 **Technical Implementation**

### **Key Functions**

#### **1. Convolutional Layer Creation**
```python
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    # Creates convolutional layer with specified parameters
    # Applies ReLU activation and optional max pooling
```

#### **2. Fully Connected Layer**
```python
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # Creates dense layer for final classification
    # Optional ReLU activation for hidden layers
```

#### **3. Prediction Function**
```python
def predict_class(images):
    # Predicts class probabilities for input images
    # Returns both predicted classes and confidence scores
```

### **Filter Visualization**
The notebook includes advanced filter visualization to understand:
- What patterns each filter detects
- How different layers respond to damage features
- Visual interpretation of learned features

### **Performance Analysis**
- **Confusion Matrix**: Shows classification accuracy breakdown
- **Error Analysis**: Examines misclassified cases
- **Validation Curves**: Tracks training vs validation performance

---

## 🎓 **Study Points for Understanding**

### **1. Convolutional Operations**
- **Convolution**: How filters slide across images to detect patterns
- **Feature Maps**: Each filter creates a feature map highlighting specific patterns
- **Hierarchical Learning**: Lower layers detect simple features, higher layers detect complex ones

### **2. Pooling Operations**
- **Max Pooling**: Reduces spatial dimensions while keeping important features
- **Translation Invariance**: Makes the network robust to small position changes
- **Parameter Reduction**: Reduces computational complexity

### **3. Activation Functions**
- **ReLU**: Introduces non-linearity, allows learning complex patterns
- **Softmax**: Converts final layer outputs to probabilities

### **4. Training Concepts**
- **Backpropagation**: How the network learns from mistakes
- **Gradient Descent**: Optimization algorithm that improves performance
- **Overfitting Prevention**: Validation monitoring and early stopping

### **5. Evaluation Metrics**
- **Accuracy**: Percentage of correctly classified images
- **Loss**: Measure of how far predictions are from actual labels
- **Confusion Matrix**: Detailed breakdown of classification performance

---

## 📂 **File Structure**
```
Vehicle_Damage_Assessment/
├── tf1_cnn.ipynb              # Main CNN implementation notebook
├── tf2_cnn.ipynb              # Alternative TF2 implementation
├── dataset.py                 # Data loading and preprocessing utilities
├── data/
│   ├── train/
│   │   ├── Fraud/             # Training fraud images
│   │   └── Non-Fraud/         # Training legitimate images
│   └── test/
│       ├── Fraud/             # Test fraud images
│       └── Non-Fraud/         # Test legitimate images
└── models/                    # Saved model checkpoints
```

---

## 🚀 **Business Impact**

### **Insurance Industry Applications**
1. **Automated Screening**: First-line defense against fraudulent claims
2. **Cost Reduction**: Reduces manual review time and costs
3. **Risk Management**: Identifies high-risk claims for detailed investigation
4. **Process Efficiency**: Accelerates legitimate claim processing

### **Performance Benefits**
- **Speed**: Instant classification of damage images
- **Consistency**: Eliminates human bias in initial screening
- **Scalability**: Can process thousands of claims simultaneously
- **Learning**: Improves over time with more data

---

## 🔬 **Advanced Features**

### **Model Interpretability**
- Filter visualization shows what the network "sees"
- Feature map analysis reveals decision-making process
- Error analysis identifies improvement opportunities

### **Robust Architecture**
- Multiple convolutional layers for hierarchical feature learning
- Dropout regularization prevents overfitting
- Validation monitoring ensures generalization

### **Production Ready**
- Batch processing for efficiency
- Model checkpointing for deployment
- Comprehensive evaluation metrics

---

## 📚 **Learning Outcomes**

After studying this implementation, you'll understand:
1. **CNN Architecture**: How convolutional layers build feature hierarchies
2. **Image Classification**: Binary classification for fraud detection
3. **Deep Learning Pipeline**: Data → Preprocessing → Training → Evaluation
4. **Model Optimization**: Training strategies and performance monitoring
5. **Business Applications**: Real-world AI applications in insurance
6. **Model Interpretability**: Understanding what the network learns

This project demonstrates a complete end-to-end deep learning solution for a real business problem, combining technical depth with practical applications.