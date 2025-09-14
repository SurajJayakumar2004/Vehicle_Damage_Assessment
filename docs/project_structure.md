# Project Structure Documentation

This document explains the organized folder structure of the Vehicle Damage Assessment project.

## 📁 Directory Structure

```
Vehicle_Damage_Assessment/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── app/                        # Web Application
│   ├── streamlit_app.py        # Main Streamlit application
│   ├── static/                 # Static assets (CSS, JS, images)
│   └── templates/              # HTML templates (if needed)
│
├── notebooks/                  # Jupyter Notebooks
│   ├── Model.ipynb            # Main training and experimentation notebook
│   └── Fraud_Detection_Model_Organized.ipynb  # Clean, organized version
│
├── models/                     # Model Files
│   ├── production/            # Production-ready models
│   │   ├── fraud_detector_optimized.h5         # Primary model (H5 format)
│   │   ├── fraud_detector_optimized.keras      # Primary model (Keras format)
│   │   └── optimal_threshold.txt               # Optimized threshold (0.60000)
│   │
│   ├── backup/                # Backup and alternative models
│   │   ├── fraud_detector_resnet50_optimized.keras  # ResNet50 backup model
│   │   ├── resnet50_optimal_threshold.txt           # ResNet50 threshold
│   │   └── fraud_detector_efficientnet.keras       # Alternative EfficientNet
│   │
│   ├── legacy/                # Deprecated/old models
│   │   ├── efficientnetv2_b0_model.*          # Original EfficientNet models
│   │   ├── resnet50_model.*                   # Original ResNet50 models
│   │   ├── convnext_tiny_model.*              # ConvNeXt models
│   │   └── ensemble_model.*                   # Ensemble experiments
│   │
│   └── artifacts/             # Training artifacts and summaries
│       ├── model_performance_summary.txt      # Performance metrics
│       ├── optimal_threshold.txt              # Threshold backup
│       └── fraud_detection_config.json       # Training configuration
│
├── config/                    # Configuration Files
│   ├── deployment_summary.json    # Deployment configuration and metrics
│   ├── training_configuration.json # Training parameters and setup
│   └── model_comparison.json      # Performance comparison data
│
├── data/                      # Dataset
│   ├── train/                 # Training data
│   │   ├── Fraud/            # Fraudulent damage images
│   │   └── Non-Fraud/        # Legitimate damage images
│   └── test/                  # Test data
│       ├── Fraud/            # Test fraudulent images
│       └── Non-Fraud/        # Test legitimate images
│
├── docs/                      # Documentation
│   └── project_structure.md   # This file
│
└── logs/                      # Logs and monitoring
    └── (training and application logs)
```

## 🎯 Directory Purposes

### `/app/` - Web Application
- **Purpose**: Contains the Streamlit web application for fraud detection
- **Key Files**: 
  - `streamlit_app.py`: Main application entry point
  - `static/`: CSS, JavaScript, and image assets
  - `templates/`: HTML templates if needed

### `/notebooks/` - Jupyter Notebooks
- **Purpose**: Contains all Jupyter notebooks for development and experimentation
- **Key Files**:
  - `Model.ipynb`: Main training notebook with all experiments
  - `Fraud_Detection_Model_Organized.ipynb`: Clean, organized version for production

### `/models/` - Model Storage
- **Purpose**: Organized storage for all model files
- **Subdirectories**:
  - `production/`: Current production models and thresholds
  - `backup/`: Alternative models ready for deployment
  - `legacy/`: Old/deprecated models for reference
  - `artifacts/`: Training summaries and performance data

### `/config/` - Configuration
- **Purpose**: Stores configuration files and deployment settings
- **Key Files**:
  - `deployment_summary.json`: Production deployment configuration
  - `training_configuration.json`: Training parameters and hyperparameters
  - `model_comparison.json`: Performance comparison data

### `/data/` - Dataset
- **Purpose**: Contains the vehicle damage image dataset
- **Structure**: Organized by train/test and Fraud/Non-Fraud categories

### `/docs/` - Documentation
- **Purpose**: Project documentation and guides
- **Contents**: Technical documentation, API references, user guides

### `/logs/` - Logging
- **Purpose**: Application and training logs for monitoring and debugging

## 🔧 Model File Naming Convention

- **Production Models**: `fraud_detector_optimized.*`
- **Backup Models**: `fraud_detector_{architecture}_optimized.*`
- **Legacy Models**: `{architecture}_model.*`
- **Thresholds**: `{model_name}_optimal_threshold.txt`

## 📊 Key Files for Deployment

1. **Production Model**: `models/production/fraud_detector_optimized.h5`
2. **Optimal Threshold**: `models/production/optimal_threshold.txt`
3. **Deployment Config**: `config/deployment_summary.json`
4. **Web App**: `app/streamlit_app.py`

## 🚀 Getting Started

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run Application**: `streamlit run app/streamlit_app.py`
3. **View Notebooks**: `jupyter notebook notebooks/`
4. **Check Configs**: Review files in `config/` directory

## 📈 Model Performance Tracking

- **Primary Metrics**: Located in `config/deployment_summary.json`
- **Detailed Analysis**: Available in `models/artifacts/`
- **Comparison Data**: Stored in `config/model_comparison.json`

This organized structure ensures:
- ✅ Clear separation of concerns
- ✅ Easy model version management
- ✅ Simplified deployment process
- ✅ Comprehensive documentation
- ✅ Scalable project architecture