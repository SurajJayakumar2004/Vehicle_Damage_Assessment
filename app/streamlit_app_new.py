import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
import io
import time

# --- Revolutionary App Config ---
st.set_page_config(
    page_title="Vehicle Damage Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme Management
# --- Session State Management ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Revolutionary CSS Design System
def get_theme_css(dark_mode=True):
    if dark_mode:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@100;200;300;400;500;600;700;800;900&display=swap');
        
        .stApp {
            background: #000000 !important;
            color: #ffffff !important;
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        .main > div {
            padding-top: 0 !important;
        }
        
        /* Apple-inspired Hero Section */
        .hero-section {
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: radial-gradient(ellipse at center, #1a1a1a 0%, #000000 100%);
            text-align: center;
            margin: -1rem -1rem 2rem -1rem;
            padding: 2rem;
        }
        
        .hero-title {
            font-size: clamp(3rem, 8vw, 6rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
            animation: fadeInUp 1s ease-out;
        }
        
        .hero-subtitle {
            font-size: clamp(1.1rem, 3vw, 1.5rem);
            font-weight: 300;
            color: #a0a0a0;
            max-width: 600px;
            margin: 0 auto 3rem auto;
            animation: fadeInUp 1s ease-out 0.2s both;
        }
        
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Tesla-inspired Feature Grid */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .feature-card {
            background: rgba(28, 28, 30, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            backdrop-filter: blur(20px);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            background: rgba(28, 28, 30, 0.95);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        .feature-number {
            display: inline-block;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #007AFF, #5856D6);
            border-radius: 50%;
            color: white;
            font-weight: 700;
            line-height: 50px;
            margin-bottom: 1.5rem;
            font-size: 1.2rem;
        }
        
        .feature-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #ffffff;
        }
        
        .feature-description {
            color: #a0a0a0;
            font-weight: 300;
            line-height: 1.6;
        }
        
        /* Revolutionary Upload Interface */
        .upload-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .upload-zone {
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 24px;
            padding: 4rem 2rem;
            text-align: center;
            background: rgba(28, 28, 30, 0.6);
            backdrop-filter: blur(20px);
            transition: all 0.3s ease;
            margin: 2rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .upload-zone::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: conic-gradient(from 0deg, transparent, rgba(0, 122, 255, 0.1), transparent);
            animation: rotate 8s linear infinite;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .upload-zone:hover::before {
            opacity: 1;
        }
        
        .upload-zone:hover {
            border-color: rgba(0, 122, 255, 0.6);
            background: rgba(28, 28, 30, 0.8);
        }
        
        @keyframes rotate {
            100% { transform: rotate(360deg); }
        }
        
        .upload-title {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #ffffff;
            position: relative;
            z-index: 1;
        }
        
        .upload-subtitle {
            color: #a0a0a0;
            font-weight: 300;
            margin-bottom: 2rem;
            position: relative;
            z-index: 1;
        }
        
        /* Tesla-inspired Results Dashboard */
        .results-section {
            background: rgba(28, 28, 30, 0.8);
            border-radius: 24px;
            padding: 3rem;
            margin: 3rem 0;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .dashboard-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            color: #ffffff;
        }
        
        .dashboard-subtitle {
            color: #a0a0a0;
            font-weight: 300;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .metric-tile {
            background: rgba(44, 44, 46, 0.8);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }
        
        .metric-tile:hover {
            transform: translateY(-4px);
            background: rgba(44, 44, 46, 1);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: #ffffff;
        }
        
        .metric-label {
            color: #a0a0a0;
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9rem;
        }
        
        /* Modern Result Items */
        .result-item {
            background: rgba(44, 44, 46, 0.6);
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.3s ease;
        }
        
        .result-item:hover {
            background: rgba(44, 44, 46, 0.8);
            transform: translateX(8px);
        }
        
        .result-info h4 {
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
            color: #ffffff;
        }
        
        .result-info p {
            color: #a0a0a0;
            font-weight: 300;
        }
        
        .result-badge {
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .badge-fraud-high {
            background: linear-gradient(135deg, #FF3B30, #FF9500);
            color: white;
        }
        
        .badge-fraud-medium {
            background: linear-gradient(135deg, #FF9500, #FFCC02);
            color: white;
        }
        
        .badge-fraud-low {
            background: linear-gradient(135deg, #FFCC02, #32D74B);
            color: white;
        }
        
        .badge-legitimate {
            background: linear-gradient(135deg, #32D74B, #30DB5B);
            color: white;
        }
        
        /* Apple-style Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #007AFF, #5856D6) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1rem 2.5rem !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            width: 100% !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 25px rgba(0, 122, 255, 0.3) !important;
            background: linear-gradient(135deg, #0056CC, #4C44C4) !important;
        }
        
        /* Theme Toggle */
        .theme-switcher {
            position: fixed;
            top: 2rem;
            right: 2rem;
            z-index: 1000;
            background: rgba(28, 28, 30, 0.9);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 50px;
            padding: 0.5rem;
        }
        
        /* Animated Result Cards - Dark Mode */
        .animated-result-card {
            background: rgba(28, 28, 30, 0.8);
            border-radius: 16px;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            cursor: pointer;
            overflow: hidden;
            backdrop-filter: blur(20px);
        }
        
        .animated-result-card:hover {
            background: rgba(28, 28, 30, 0.95);
            transform: translateY(-4px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.3);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .result-item-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem;
        }
        
        .expand-arrow {
            color: #a0a0a0;
            font-size: 1.2rem;
            transition: all 0.3s ease;
            margin-left: 1rem;
        }
        
        .result-details {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding: 0;
            overflow: hidden;
        }
        
        .details-grid {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 2rem;
            padding: 2rem;
        }
        
        @media (max-width: 768px) {
            .details-grid {
                grid-template-columns: 1fr;
                gap: 1rem;
            }
        }
        
        .image-section {
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(44, 44, 46, 0.5);
            border-radius: 12px;
            padding: 2rem;
            min-height: 200px;
        }
        
        .image-placeholder {
            text-align: center;
            color: #a0a0a0;
            font-size: 1.2rem;
        }
        
        .filename-text {
            display: block;
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #ffffff;
        }
        
        .analysis-section h5 {
            color: #ffffff;
            margin: 1.5rem 0 0.8rem 0;
            font-size: 1.1rem;
        }
        
        .analysis-section h5:first-child {
            margin-top: 0;
        }
        
        .analysis-section ul {
            list-style: none;
            padding: 0;
            margin: 0 0 1.5rem 0;
        }
        
        .analysis-section li {
            color: #d1d1d6;
            margin: 0.5rem 0;
            padding-left: 1rem;
            position: relative;
        }
        
        .analysis-section li::before {
            content: '•';
            color: #007AFF;
            position: absolute;
            left: 0;
        }
        
        .risk-interpretation {
            color: #d1d1d6;
            line-height: 1.6;
            background: rgba(44, 44, 46, 0.5);
            padding: 1rem;
            border-radius: 8px;
            margin: 0;
        }
        
        /* Results Header - Dark Mode */
        .results-header {
            text-align: center;
            margin: 2rem 0;
            padding: 2rem;
            background: rgba(28, 28, 30, 0.6);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .results-header h2 {
            color: #ffffff;
            margin-bottom: 0.5rem;
            font-size: 1.8rem;
        }
        
        .results-header p {
            color: #a0a0a0;
            font-size: 1rem;
            margin: 0;
        }
        
        /* Result Summary Cards - Dark Mode */
        .result-summary-card {
            background: rgba(28, 28, 30, 0.8);
            border-radius: 12px;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            backdrop-filter: blur(20px);
        }
        
        .result-summary-card:hover {
            background: rgba(28, 28, 30, 0.95);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }
        
        .summary-content {
            display: flex;
            align-items: center;
            padding: 1rem 1.5rem;
            gap: 1rem;
        }
        
        .summary-icon {
            font-size: 1.2rem;
        }
        
        .summary-filename {
            flex: 1;
            color: #ffffff;
            font-weight: 500;
        }
        
        .summary-confidence {
            color: #a0a0a0;
            font-size: 0.9rem;
        }
        
        .summary-status {
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .summary-content-compact {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            text-align: center;
            padding: 1rem;
        }
        
        .summary-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .summary-content-compact .summary-filename {
            font-size: 0.95rem;
            word-break: break-word;
        }
        
        .summary-content-compact .summary-confidence {
            background: rgba(74, 144, 226, 0.1);
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            white-space: nowrap;
        }
        
        .summary-content-expanded {
            display: flex;
            align-items: center;
            gap: 2rem;
            padding: 1.5rem;
        }
        
        .summary-left {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .summary-right {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .summary-header-expanded {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .summary-filename-large {
            font-size: 1.2rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        .summary-details {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .summary-confidence-large {
            background: rgba(74, 144, 226, 0.2);
            color: #4A90E2;
            padding: 0.4rem 1rem;
            border-radius: 15px;
            font-size: 0.95rem;
            font-weight: 500;
        }
        
        .summary-status-large {
            padding: 0.4rem 1rem;
            border-radius: 15px;
            font-size: 0.95rem;
            font-weight: 600;
        }
        
        .summary-quick-stats {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .quick-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #a0a0a0;
            margin-bottom: 0.25rem;
        }
        
        .stat-value {
            font-size: 1rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        /* Full-width Summary Card - Dark Mode */
        .result-summary-card-fullwidth {
            background: rgba(28, 28, 30, 0.8);
            border-radius: 16px;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            backdrop-filter: blur(20px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        .result-summary-card-fullwidth:hover {
            background: rgba(28, 28, 30, 0.95);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
        }
        
        .summary-content-fullwidth {
            padding: 2rem;
        }
        
        .summary-main-info {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .summary-header-fullwidth {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        
        .summary-filename-fullwidth {
            font-size: 1.4rem;
            font-weight: 600;
            color: #ffffff;
            flex: 1;
            min-width: 200px;
        }
        
        .summary-confidence-badge {
            background: rgba(74, 144, 226, 0.2);
            color: #4A90E2;
            padding: 0.5rem 1.2rem;
            border-radius: 20px;
            font-size: 1rem;
            font-weight: 500;
        }
        
        .summary-status-badge {
            padding: 0.5rem 1.2rem;
            border-radius: 20px;
            font-size: 1rem;
            font-weight: 600;
        }
        
        .summary-quick-metrics {
            display: flex;
            gap: 2rem;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .metric-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .metric-label {
            font-size: 1rem;
            color: #a0a0a0;
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 1rem;
            font-weight: 600;
            color: #ffffff;
        }
        
        .badge-fraud-high .summary-status {
            background: linear-gradient(135deg, #FF3B30, #FF9500);
            color: white;
        }
        
        .badge-fraud-medium .summary-status {
            background: linear-gradient(135deg, #FF9500, #FFCC02);
            color: white;
        }
        
        .badge-fraud-low .summary-status {
            background: linear-gradient(135deg, #FFCC02, #32D74B);
            color: white;
        }
        
        .badge-legitimate .summary-status {
            background: linear-gradient(135deg, #32D74B, #30DB5B);
            color: white;
        }
        
        /* Animation keyframes */
        @keyframes slideDown {
            from {
                opacity: 0;
                max-height: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                max-height: 500px;
                transform: translateY(0);
            }
        }
        
        @keyframes slideUp {
            from {
                opacity: 1;
                max-height: 500px;
                transform: translateY(0);
            }
            to {
                opacity: 0;
                max-height: 0;
                transform: translateY(-10px);
            }
        }

        /* Hide Streamlit UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        header {visibility: hidden;}
        .stApp > header {background: transparent;}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 4px;
        }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@100;200;300;400;500;600;700;800;900&display=swap');
        
        .stApp {
            background: #ffffff !important;
            color: #1d1d1f !important;
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        .main > div {
            padding-top: 0 !important;
        }
        
        /* Light Mode Hero Section */
        .hero-section {
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: radial-gradient(ellipse at center, #f5f5f7 0%, #ffffff 100%);
            text-align: center;
            margin: -1rem -1rem 2rem -1rem;
            padding: 2rem;
        }
        
        .hero-title {
            font-size: clamp(3rem, 8vw, 6rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(135deg, #1d1d1f 0%, #86868b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
            animation: fadeInUp 1s ease-out;
        }
        
        .hero-subtitle {
            font-size: clamp(1.1rem, 3vw, 1.5rem);
            font-weight: 300;
            color: #515151;
            max-width: 600px;
            margin: 0 auto 3rem auto;
            animation: fadeInUp 1s ease-out 0.2s both;
        }
        
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Light Mode Feature Grid */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            backdrop-filter: blur(20px);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(0, 122, 255, 0.3), transparent);
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            background: rgba(255, 255, 255, 1);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            border-color: rgba(0, 122, 255, 0.2);
        }
        
        .feature-number {
            display: inline-block;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #007AFF, #5856D6);
            border-radius: 50%;
            color: white;
            font-weight: 700;
            line-height: 50px;
            margin-bottom: 1.5rem;
            font-size: 1.2rem;
        }
        
        .feature-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1d1d1f;
        }
        
        .feature-description {
            color: #515151;
            font-weight: 300;
            line-height: 1.6;
        }
        
        /* Light Mode Upload Interface */
        .upload-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .upload-zone {
            border: 2px dashed rgba(0, 0, 0, 0.2);
            border-radius: 24px;
            padding: 4rem 2rem;
            text-align: center;
            background: rgba(245, 245, 247, 0.8);
            backdrop-filter: blur(20px);
            transition: all 0.3s ease;
            margin: 2rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .upload-zone::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: conic-gradient(from 0deg, transparent, rgba(0, 122, 255, 0.1), transparent);
            animation: rotate 8s linear infinite;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .upload-zone:hover::before {
            opacity: 1;
        }
        
        .upload-zone:hover {
            border-color: rgba(0, 122, 255, 0.4);
            background: rgba(245, 245, 247, 1);
        }
        
        @keyframes rotate {
            100% { transform: rotate(360deg); }
        }
        
        .upload-title {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #1d1d1f;
            position: relative;
            z-index: 1;
        }
        
        .upload-subtitle {
            color: #515151;
            font-weight: 400;
            margin-bottom: 2rem;
            position: relative;
            z-index: 1;
        }
        
        /* Light mode text visibility fixes */
        .stMarkdown, .stText {
            color: #1d1d1f !important;
        }
        
        .stInfo {
            background-color: rgba(0, 122, 255, 0.1) !important;
            color: #1d1d1f !important;
            border: 1px solid rgba(0, 122, 255, 0.2) !important;
        }
        
        /* Ready section styling for light mode */
        .ready-section {
            text-align: center;
            padding: 2rem;
            color: #515151;
        }
        
        .ready-section h3 {
            color: #1d1d1f;
            margin-bottom: 0.5rem;
        }
        
        .ready-section p {
            color: #515151;
        }
        
        /* Override any remaining light colors in light mode */
        p, span, div {
            color: inherit;
        }
        
        /* Ensure proper contrast for all text elements */
        .stMarkdown p, .stMarkdown span, .stMarkdown div {
            color: #1d1d1f !important;
        }
        
        /* Fix for specific Streamlit components */
        .stSelectbox label, .stFileUploader label, .stButton label {
            color: #1d1d1f !important;
        }
        
        /* Comprehensive text color fixes for light mode */
        .stMarkdown p, .stText, .stMarkdown li, .stMarkdown span {
            color: #1d1d1f !important;
        }
        
        .stExpander .streamlit-expanderHeader {
            color: #1d1d1f !important;
        }
        
        .stExpander .streamlit-expanderContent p,
        .stExpander .streamlit-expanderContent li,
        .stExpander .streamlit-expanderContent span {
            color: #1d1d1f !important;
        }
        
        /* Animated Result Cards - Light Mode */
        .animated-result-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            margin: 1rem 0;
            border: 1px solid rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            cursor: pointer;
            overflow: hidden;
            backdrop-filter: blur(20px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        
        .animated-result-card:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateY(-4px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.1);
            border-color: rgba(0, 122, 255, 0.2);
        }
        
        .result-item-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem;
        }
        
        .expand-arrow {
            color: #515151;
            font-size: 1.2rem;
            transition: all 0.3s ease;
            margin-left: 1rem;
        }
        
        .result-details {
            border-top: 1px solid rgba(0, 0, 0, 0.08);
            padding: 0;
            overflow: hidden;
        }
        
        .details-grid {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 2rem;
            padding: 2rem;
        }
        
        .image-section {
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(245, 245, 247, 0.8);
            border-radius: 12px;
            padding: 2rem;
            min-height: 200px;
        }
        
        .image-placeholder {
            text-align: center;
            color: #515151;
            font-size: 1.2rem;
        }
        
        .filename-text {
            display: block;
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #1d1d1f;
        }
        
        .analysis-section h5 {
            color: #1d1d1f;
            margin: 1.5rem 0 0.8rem 0;
            font-size: 1.1rem;
        }
        
        .analysis-section h5:first-child {
            margin-top: 0;
        }
        
        .analysis-section ul {
            list-style: none;
            padding: 0;
            margin: 0 0 1.5rem 0;
        }
        
        .analysis-section li {
            color: #515151;
            margin: 0.5rem 0;
            padding-left: 1rem;
            position: relative;
        }
        
        .analysis-section li::before {
            content: '•';
            color: #007AFF;
            position: absolute;
            left: 0;
        }
        
        .risk-interpretation {
            color: #515151;
            line-height: 1.6;
            background: rgba(245, 245, 247, 0.8);
            padding: 1rem;
            border-radius: 8px;
            margin: 0;
        }
        
        /* Results Header - Light Mode */
        .results-header {
            text-align: center;
            margin: 2rem 0;
            padding: 2rem;
            background: rgba(245, 245, 247, 0.8);
            border-radius: 16px;
            border: 1px solid rgba(0, 0, 0, 0.08);
        }
        
        .results-header h2 {
            color: #1d1d1f;
            margin-bottom: 0.5rem;
            font-size: 1.8rem;
        }
        
        .results-header p {
            color: #515151;
            font-size: 1rem;
            margin: 0;
        }
        
        /* Result Summary Cards - Light Mode */
        .result-summary-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            margin: 0.5rem 0;
            border: 1px solid rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            backdrop-filter: blur(20px);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        }
        
        .result-summary-card:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        }
        
        .summary-content {
            display: flex;
            align-items: center;
            padding: 1rem 1.5rem;
            gap: 1rem;
        }
        
        .summary-icon {
            font-size: 1.2rem;
        }
        
        .summary-filename {
            flex: 1;
            color: #1d1d1f;
            font-weight: 500;
        }
        
        .summary-confidence {
            color: #515151;
            font-size: 0.9rem;
        }
        
        .summary-status {
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .summary-content-compact {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            text-align: center;
            padding: 1rem;
        }
        
        .summary-header {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .summary-content-expanded {
            display: flex;
            align-items: center;
            gap: 2rem;
            padding: 1.5rem;
        }
        
        .summary-left {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .summary-right {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .summary-header-expanded {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .summary-filename-large {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        .summary-details {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .summary-confidence-large {
            background: rgba(74, 144, 226, 0.1);
            color: #4A90E2;
            padding: 0.4rem 1rem;
            border-radius: 15px;
            font-size: 0.95rem;
            font-weight: 500;
        }
        
        .summary-status-large {
            padding: 0.4rem 1rem;
            border-radius: 15px;
            font-size: 0.95rem;
            font-weight: 600;
        }
        
        .summary-quick-stats {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .quick-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #515151;
            margin-bottom: 0.25rem;
        }
        
        .stat-value {
            font-size: 1rem;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        /* Full-width Summary Card - Light Mode */
        .result-summary-card-fullwidth {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            margin: 1rem 0;
            border: 1px solid rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            backdrop-filter: blur(20px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        }
        
        .result-summary-card-fullwidth:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        }
        
        .summary-content-fullwidth {
            padding: 2rem;
        }
        
        .summary-main-info {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .summary-header-fullwidth {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        
        .summary-filename-fullwidth {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1d1d1f;
            flex: 1;
            min-width: 200px;
        }
        
        .summary-confidence-badge {
            background: rgba(74, 144, 226, 0.1);
            color: #4A90E2;
            padding: 0.5rem 1.2rem;
            border-radius: 20px;
            font-size: 1rem;
            font-weight: 500;
        }
        
        .summary-status-badge {
            padding: 0.5rem 1.2rem;
            border-radius: 20px;
            font-size: 1rem;
            font-weight: 600;
        }
        
        .summary-quick-metrics {
            display: flex;
            gap: 2rem;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .metric-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .metric-label {
            font-size: 1rem;
            color: #515151;
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 1rem;
            font-weight: 600;
            color: #1d1d1f;
        }
        
        /* Light Mode Results Dashboard */
        .results-section {
            background: rgba(245, 245, 247, 0.9);
            border-radius: 24px;
            padding: 3rem;
            margin: 3rem 0;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 0, 0, 0.08);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        }
        
        .dashboard-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            color: #1d1d1f;
        }
        
        .dashboard-subtitle {
            color: #515151;
            font-weight: 300;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .metric-tile {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(0, 0, 0, 0.08);
            transition: transform 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        
        .metric-tile:hover {
            transform: translateY(-4px);
            background: rgba(255, 255, 255, 1);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: #1d1d1f;
        }
        
        .metric-label {
            color: #515151;
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9rem;
        }
        
        /* Light Mode Result Items */
        .result-item {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(0, 0, 0, 0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        }
        
        .result-item:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateX(8px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        }
        
        .result-info h4 {
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
            color: #1d1d1f;
        }
        
        .result-info p {
            color: #515151;
            font-weight: 300;
        }
        
        .result-badge {
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .badge-fraud-high {
            background: linear-gradient(135deg, #FF3B30, #FF9500);
            color: white;
        }
        
        .badge-fraud-medium {
            background: linear-gradient(135deg, #FF9500, #FFCC02);
            color: white;
        }
        
        .badge-fraud-low {
            background: linear-gradient(135deg, #FFCC02, #32D74B);
            color: white;
        }
        
        .badge-legitimate {
            background: linear-gradient(135deg, #32D74B, #30DB5B);
            color: white;
        }
        
        /* Light Mode Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #007AFF, #5856D6) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1rem 2.5rem !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            width: 100% !important;
            box-shadow: 0 4px 15px rgba(0, 122, 255, 0.2) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 25px rgba(0, 122, 255, 0.3) !important;
            background: linear-gradient(135deg, #0056CC, #4C44C4) !important;
        }
        
        /* Secondary buttons in light mode */
        .stButton > button[kind="secondary"] {
            background: rgba(242, 242, 247, 0.8) !important;
            color: #1d1d1f !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: rgba(242, 242, 247, 1) !important;
            color: #1d1d1f !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Theme Toggle Light Mode */
        .theme-switcher {
            position: fixed;
            top: 2rem;
            right: 2rem;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 50px;
            padding: 0.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        
        /* Hide Streamlit UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        header {visibility: hidden;}
        .stApp > header {background: transparent;}
        
        /* Custom Scrollbar Light Mode */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.05);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 0, 0, 0.3);
        }
        </style>
        """

# Apply CSS
st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Theme Toggle
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    theme_text = "Light Mode" if st.session_state.dark_mode else "Dark Mode"
    if st.button(theme_text, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.experimental_rerun()

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">Vehicle Damage<br>Fraud Detector</div>
    <div class="hero-subtitle">Advanced AI-powered fraud detection for vehicle damage claims using cutting-edge machine learning technology</div>
</div>
""", unsafe_allow_html=True)

# Feature Grid
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-number">1</div>
        <div class="feature-title">Upload Images</div>
        <div class="feature-description">Upload vehicle damage photos with support for multiple formats and batch processing</div>
    </div>
    <div class="feature-card">
        <div class="feature-number">2</div>
        <div class="feature-title">AI Analysis</div>
        <div class="feature-description">Advanced neural networks analyze images for fraud indicators and suspicious patterns</div>
    </div>
    <div class="feature-card">
        <div class="feature-number">3</div>
        <div class="feature-title">Instant Results</div>
        <div class="feature-description">Get real-time fraud risk assessment with detailed confidence scores and evidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload Section
# Create a narrower upload section
upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])

with upload_col2:
    st.markdown("### 📤 Upload Vehicle Damage Images")
    st.markdown("*Upload clear photos of vehicle damage for AI-powered fraud analysis*")

    # Upload section with better alignment
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload clear photos of vehicle damage for best results",
        key="file_uploader"
    )

    # Clear button in a centered container within the narrow section
    clear_col1, clear_col2, clear_col3 = st.columns([1, 2, 1])
    with clear_col2:
        if st.button("🗑️ Clear All Files", type="secondary", help="Clear uploaded files and results", use_container_width=True):
            st.session_state.uploaded_files = None
            st.session_state.analysis_results = None
            st.session_state.show_results = False
            st.rerun()

# Update session state with current uploads
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# --- Model Loading Functions ---
@st.cache_resource
def load_model_only():
    """Load only the model (cached)"""
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "production", "fraud_detector_optimized.h5")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.info("Please ensure the model files are in the correct location.")
        return None
    
    model = tf.keras.models.load_model(model_path)
    return model

def get_current_threshold():
    """Get the current threshold (not cached - updates automatically)"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    threshold_path = os.path.join(project_root, "models", "production", "optimal_threshold.txt")
    
    try:
        # Get file modification time for display
        mod_time = os.path.getmtime(threshold_path)
        mod_time_str = pd.Timestamp(mod_time, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
            
        return threshold, mod_time_str
    except Exception as e:
        st.warning(f"Could not read threshold file: {e}")
        return 0.5, "Unknown"

def load_optimized_model():
    """Load model and get current threshold"""
    model = load_model_only()
    threshold, mod_time = get_current_threshold()
    return model, threshold, mod_time

# Show uploaded files if any exist
current_files = st.session_state.uploaded_files or uploaded_files

if current_files:
    # Show image previews
    cols = st.columns(min(4, len(current_files)))
    for idx, file in enumerate(current_files):
        try:
            file.seek(0)  # Reset file pointer
            img = Image.open(file)
            cols[idx % 4].image(img, caption=file.name, use_column_width=True)
        except:
            cols[idx % 4].write(f"📄 {file.name}")
    
    if st.button("🔍 Analyze Images for Fraud", type="primary"):
        with st.spinner("🤖 AI is analyzing your images..."):
            model, threshold, mod_time = load_optimized_model()
            
            if model is None:
                st.error("❌ Unable to load the AI model. Please try again.")
                st.stop()
            
            # Real analysis with actual model
            results = []
            files_to_analyze = st.session_state.uploaded_files or uploaded_files
            for file in files_to_analyze:
                try:
                    # Reset file pointer to beginning
                    file.seek(0)
                    
                    # Load and preprocess image properly
                    img = Image.open(file)
                    img = img.convert("RGB")
                    img = img.resize((256, 256), resample=Image.BILINEAR)
                    
                    # Convert to numpy array with proper dtype and normalization
                    arr = np.array(img).astype("float32") / 255.0
                    arr = np.expand_dims(arr, axis=0)
                    
                    # Defensive check for shape
                    if arr.shape != (1, 256, 256, 3):
                        st.warning(f"Image {file.name} has unexpected shape {arr.shape}. Skipping.")
                        continue
                    
                    # Get prediction probability from optimized model
                    pred_probability = float(model.predict(arr, verbose=0)[0][0])
                    
                    # Apply optimized threshold
                    is_fraud = pred_probability >= threshold
                    
                    # Calculate confidence
                    confidence = pred_probability if is_fraud else 1.0 - pred_probability
                    
                    results.append({
                        "filename": file.name,
                        "probability": pred_probability,
                        "is_fraud": is_fraud,
                        "confidence": confidence
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue
            
            # Store results in session state
            st.session_state.analysis_results = results
            st.session_state.show_results = True
            
            # Results Dashboard
            fraud_count = sum(1 for r in results if r["is_fraud"])
            total_count = len(results)
            
            st.markdown(f"""
            <div class="results-header">
                <h2>📊 Analysis Complete</h2>
                <p>Analyzed {total_count} images • {fraud_count} fraud detected • {total_count - fraud_count} legitimate</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual Results with Animated Cards
            for i, result in enumerate(results):
                prob = result["probability"]
                is_fraud = result["is_fraud"]
                
                if is_fraud:
                    if prob > 0.8:
                        badge_class = "badge-fraud-high"
                        status = "High Risk"
                        status_icon = "🔴"
                    elif prob > 0.6:
                        badge_class = "badge-fraud-medium"
                        status = "Medium Risk"
                        status_icon = "🟡"
                    else:
                        badge_class = "badge-fraud-low"
                        status = "Low Risk"
                        status_icon = "🟠"
                else:
                    badge_class = "badge-legitimate"
                    status = "Legitimate"
                    status_icon = "✅"
                
                # Create unique key for each result card
                card_key = f"result_card_{i}_{result['filename']}"
                
                # Full-width summary card with embedded expandable details
                st.markdown(f"""
                <div class="result-summary-card-fullwidth {badge_class}">
                    <div class="summary-content-fullwidth">
                        <div class="summary-main-info">
                            <div class="summary-header-fullwidth">
                                <span class="summary-icon">{status_icon}</span>
                                <span class="summary-filename-fullwidth">{result["filename"]}</span>
                                <span class="summary-confidence-badge">{result["confidence"]:.1%} confidence</span>
                                <span class="summary-status-badge">{status}</span>
                            </div>
                            <div class="summary-quick-metrics">
                                <div class="metric-item">
                                    <span class="metric-label">Probability:</span>
                                    <span class="metric-value">{result['probability']:.3f}</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">Decision:</span>
                                    <span class="metric-value">{'⚠️ Fraud Detected' if result['is_fraud'] else '✅ Legitimate'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Embedded expandable details inside the summary context
                with st.expander(f"📋 View Detailed Analysis for {result['filename']}", expanded=False):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Show image if available
                        try:
                            for file in files_to_analyze:
                                if file.name == result["filename"]:
                                    file.seek(0)
                                    img = Image.open(file) 
                                    st.image(img, caption=result["filename"], use_column_width=True)
                                    break
                        except:
                            st.markdown("""
                            <div style="text-align: center; padding: 1rem; background: rgba(100,100,100,0.1); border-radius: 8px;">
                                🖼️<br><small>Image preview not available</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        **🎯 Analysis Results:**
                        - **Fraud Probability:** {result['probability']:.4f}
                        - **Confidence Level:** {result['confidence']:.1%}
                        - **Risk Assessment:** {status}
                        - **Final Decision:** {'⚠️ Fraud Detected' if result['is_fraud'] else '✅ Legitimate Claim'}
                        
                        **🔧 Technical Details:**
                        - **Model Threshold:** {threshold:.3f}
                        - **Analysis Time:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                        - **Model Version:** EfficientNetV2-B0
                        
                        **📊 Risk Interpretation:**
                        
                        {'High probability of fraudulent activity. Requires immediate investigation.' if prob > 0.8 else
                         'Moderate fraud risk detected. Manual review recommended.' if prob > 0.6 and is_fraud else
                         'Low fraud risk but above threshold. Consider additional verification.' if is_fraud else
                         'Legitimate claim with high confidence. Proceed with normal processing.'}
                        """)
# Display stored results if they exist (handles theme switching)
elif st.session_state.show_results and st.session_state.analysis_results:
    results = st.session_state.analysis_results
    fraud_count = sum(1 for r in results if r["is_fraud"])
    total_count = len(results)
    
    # Results Header
    st.markdown(f"""
    <div class="results-header">
        <h2>📊 Previous Analysis Results</h2>
        <p>Analyzed {total_count} images • {fraud_count} fraud detected • {total_count - fraud_count} legitimate</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual Results with Animated Cards
    for i, result in enumerate(results):
        prob = result["probability"]
        is_fraud = result["is_fraud"]
        
        if is_fraud:
            if prob > 0.8:
                badge_class = "badge-fraud-high"
                status = "High Risk"
                status_icon = "🔴"
            elif prob > 0.6:
                badge_class = "badge-fraud-medium"
                status = "Medium Risk"
                status_icon = "🟡"
            else:
                badge_class = "badge-fraud-low"
                status = "Low Risk"
                status_icon = "🟠"
        else:
            badge_class = "badge-legitimate"
            status = "Legitimate"
            status_icon = "✅"
        
        # Create unique key for each stored result card
        card_key = f"stored_result_card_{i}_{result['filename']}"
        
        # Streamlit native expandable card for stored results
        with st.expander(f"{status_icon} **{result['filename']}** - {status} ({result['confidence']:.1%} confidence)", expanded=False):
            st.markdown(f"""
            **🎯 Previous Analysis Results:**
            - **Fraud Probability:** {result['probability']:.4f}
            - **Confidence Level:** {result['confidence']:.1%}
            - **Risk Assessment:** {status}
            - **Final Decision:** {'⚠️ Fraud Detected' if result['is_fraud'] else '✅ Legitimate Claim'}
            
            **📊 Risk Interpretation:**
            
            {'High probability of fraudulent activity. Requires immediate investigation.' if prob > 0.8 else
             'Moderate fraud risk detected. Manual review recommended.' if prob > 0.6 and is_fraud else
             'Low fraud risk but above threshold. Consider additional verification.' if is_fraud else
             'Legitimate claim with high confidence. Proceed with normal processing.'}
            """)
        
        # Summary card for stored results
        st.markdown(f"""
        <div class="result-summary-card {badge_class}">
            <div class="summary-content">
                <span class="summary-icon">{status_icon}</span>
                <span class="summary-filename">{result["filename"]}</span>
                <span class="summary-confidence">{result["confidence"]:.1%}</span>
                <span class="summary-status">{status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-container">
        <div class="ready-section">
            <h3>Ready to Get Started?</h3>
            <p>Upload your vehicle damage images above to begin the AI-powered fraud analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)