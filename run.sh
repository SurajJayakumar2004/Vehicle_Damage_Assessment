#!/bin/bash

# Vehicle Damage Assessment - Startup Script
echo "🚗 Vehicle Damage Assessment - CNN Fraud Detection System"
echo "========================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Check if models exist
if [ ! -f "models/vehicle_damage_model.keras" ] && [ ! -f "models/vehicle_damage_cnn_model.h5" ]; then
    echo "⚠️  No trained model found!"
    echo "   Please run the training notebook first: notebooks/vehicle_damage_cnn.ipynb"
    echo "   Or download pre-trained models from the releases page."
    exit 1
fi

# Start the application
echo "🚀 Starting Vehicle Damage Assessment Web App..."
echo "   Opening in browser: http://localhost:8501"
echo "   Press Ctrl+C to stop the application"
echo ""

cd src
streamlit run streamlit_app.py
