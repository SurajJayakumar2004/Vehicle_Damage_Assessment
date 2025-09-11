# Vehicle Damage Assessment - AI Fraud Detection System

## 🎯 Project Objective

Develop an intelligent deep learning system to classify vehicle damage images as **Fraud** or **Non-Fraud** for insurance claim assessment, targeting **≥93.4% accuracy** using advanced machine learning techniques.

## 📊 Dataset Overview

- **Total Images**: 6,616 vehicle damage photographs
- **Classes**: Binary classification (Fraud vs Non-Fraud)
- **Challenge**: Highly imbalanced dataset (~1:25 ratio)
- **Format**: JPG images of varying resolutions
- **Location**: `data/` directory (protected)

## 🏗️ Recommended Model Architectures

### Transfer Learning Approach (Recommended)

#### **EfficientNetV2-B0** ⭐ Primary Choice
- **Why**: Best performance-to-efficiency ratio
- **Strengths**: Optimized architecture, fast training, excellent accuracy
- **Expected Performance**: 94-96%
- **Training Time**: 30-45 minutes

#### **ResNet50** 🛡️ Reliable Alternative  
- **Why**: Proven architecture with robust performance
- **Strengths**: Well-established, extensively researched
- **Expected Performance**: 92-94%
- **Training Time**: 45-60 minutes

#### **ConvNeXt-Tiny** 🚀 Modern Architecture
- **Why**: State-of-the-art design principles
- **Strengths**: Contemporary techniques, competitive results
- **Expected Performance**: 93-95%
- **Training Time**: 35-50 minutes

### Ensemble Strategy (Advanced)
- **Concept**: Combine all three models for superior performance
- **Method**: Weighted voting based on individual model strength
- **Expected Performance**: 95-97%
- **Benefit**: Increased robustness and reliability

## ⚖️ Class Imbalance Solutions

### The Core Challenge
- **Fraud samples**: ~200-500 images (minority class)
- **Non-fraud samples**: ~5,000-6,000 images (majority class)
- **Impact**: Standard training heavily biases toward majority class

### Solution 1: Oversampling with WeightedRandomSampler
**Concept**: Balance training exposure without duplicating data
- Calculate inverse frequency weights for each class
- Oversample minority class during each training epoch
- Ensure equal representation of both fraud and non-fraud cases
- **Benefit**: Natural balancing without artificial data creation

### Solution 2: Undersampling Majority Class
**Concept**: Reduce dataset size while maintaining balance
- Strategically select representative subset of non-fraud images
- Preserve challenging cases and edge examples
- Create more manageable, balanced training set
- **Benefit**: Faster training with improved class balance

### Solution 3: Targeted Data Augmentation
**Concept**: Generate synthetic fraud samples through augmentation
- Apply intensive augmentation **only** to fraud images
- Use rotation, zoom, brightness changes, noise addition
- Light augmentation for non-fraud to prevent over-representation
- **Techniques**: Rotation, scaling, color jittering, noise injection
- **Benefit**: Increases minority class samples while preserving quality

### Solution 4: Class-Weighted Cross-Entropy Loss
**Concept**: Penalize fraud misclassification more heavily
- Pass `class_weight` parameter to loss function during training
- Assign higher penalty for incorrectly classified fraud cases
- Automatically calculated based on class frequency distribution
- **Implementation**: Built into TensorFlow/Keras training loop
- **Benefit**: Forces model to prioritize fraud detection accuracy

### Solution 5: Advanced Loss Functions
**Concept**: Focus learning on difficult examples
- Implement Focal Loss for hard example mining
- Reduce contribution from easily classified samples
- Concentrate training effort on challenging fraud cases
- **Parameters**: Alpha (class weighting), Gamma (difficulty focus)
- **Benefit**: Superior performance on edge cases and difficult fraud scenarios

## 🚀 Implementation Strategy

### Phase 1: Environment & Baseline
1. **Setup**: Install TensorFlow 2.13+, required dependencies
2. **Data Pipeline**: Implement efficient image loading and preprocessing  
3. **Baseline Model**: Train simple CNN for performance benchmark (85-90%)
4. **Evaluation Framework**: Establish metrics tracking and validation

### Phase 2: Transfer Learning Implementation
1. **Model Selection**: Start with EfficientNetV2-B0
2. **Fine-tuning Strategy**: Freeze base layers initially, gradual unfreezing
3. **Class Balancing**: Implement WeightedRandomSampler from start
4. **Performance Target**: Achieve 93-96% accuracy

### Phase 3: Advanced Techniques
1. **Loss Function**: Implement class-weighted cross-entropy
2. **Data Augmentation**: Apply targeted augmentation for fraud images
3. **Alternative Models**: Train ResNet50 and ConvNeXt-Tiny
4. **Optimization**: Hyperparameter tuning and learning rate scheduling

### Phase 4: Ensemble Development  
1. **Model Combination**: Integrate best-performing individual models
2. **Voting Strategy**: Implement weighted averaging mechanism
3. **Weight Optimization**: Determine optimal model contribution weights
4. **Final Validation**: Achieve target 95-97% ensemble accuracy

## 📈 Success Metrics & Evaluation

### Primary Metrics
- **Accuracy**: ≥93.4% (project requirement)
- **Fraud Recall**: ≥90% (critical - minimize missed fraud cases)
- **Precision**: Maintain balance to control false positives
- **F1-Score**: Overall performance indicator

### Business Impact Metrics
- **False Negative Rate**: Minimize undetected fraud (business critical)
- **False Positive Rate**: Control legitimate claims flagged as fraud
- **Processing Efficiency**: Enable automated preliminary screening
- **Cost Reduction**: Reduce manual review requirements

### Technical Performance
- **Training Time**: <60 minutes per model
- **Model Size**: <100MB for deployment
- **Inference Speed**: Real-time prediction capability
- **Resource Usage**: Efficient memory and compute utilization

## 🎯 Expected Outcomes

### Performance Progression
1. **Baseline CNN**: 85-90% accuracy (validation concept)
2. **Single Transfer Learning**: 93-96% accuracy (meet requirements)
3. **Optimized Single Model**: 94-96% accuracy (with class balancing)
4. **Ensemble Method**: 95-97% accuracy (exceed requirements)

### Key Success Factors
- **Data Quality**: Proper preprocessing and augmentation
- **Class Balance**: Effective imbalance handling techniques
- **Model Selection**: Optimal architecture for the problem
- **Evaluation Rigor**: Comprehensive performance assessment

## 📋 Project Structure

```
Vehicle_Damage_Assessment/
├── README.md                    # This comprehensive guide
├── data/                        # Protected dataset (6,616 images)
│   ├── train/
│   │   ├── Fraud/               # ~200 fraud cases
│   │   └── Non-Fraud/           # ~5,000 legitimate cases
│   └── test/
│       ├── Fraud/               # Fraud test set
│       └── Non-Fraud/           # Legitimate test set
└── [Implementation files to be created]
```

## 💡 Key Implementation Notes

### Critical Considerations
1. **Data Protection**: Never modify or delete images in `data/` directory
2. **Class Imbalance**: Address from project start, not as afterthought
3. **Validation Strategy**: Use stratified splits to maintain class ratios
4. **Metric Focus**: Prioritize fraud recall over overall accuracy when needed

### Recommended Technology Stack
- **Framework**: TensorFlow 2.13+ (Apple Silicon compatible)
- **Image Processing**: OpenCV, PIL for preprocessing
- **Visualization**: Matplotlib, Seaborn for analysis
- **Metrics**: Scikit-learn for comprehensive evaluation
- **Development**: Jupyter notebooks for experimentation

### Quality Assurance
- **Cross-Validation**: Implement k-fold validation for robust assessment
- **Error Analysis**: Analyze misclassified cases for insights
- **Performance Monitoring**: Track training curves and validation metrics
- **Model Comparison**: Systematic comparison across architectures

---

## 🎯 Summary

This project implements a state-of-the-art fraud detection system using transfer learning with EfficientNetV2-B0, ResNet50, and ConvNeXt-Tiny architectures. The approach specifically addresses class imbalance through WeightedRandomSampler, targeted data augmentation, and class-weighted cross-entropy loss. An ensemble method combining all three models targets 95-97% accuracy for robust, production-ready fraud detection.

**Status**: Ready for implementation with comprehensive strategy and protected dataset (6,616 images).

*Last Updated: September 11, 2025*