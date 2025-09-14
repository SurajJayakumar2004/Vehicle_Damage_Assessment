import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
import io
import time

# --- App Config ---
st.set_page_config(
    page_title="Vehicle Damage Fraud Detector",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "AI-powered fraud detection for vehicle damage claims. Simply upload images and get instant analysis!"
    }
)

# --- Theme Management ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# --- Revolutionary Design System ---
def get_theme_css(dark_mode=False):
    if dark_mode:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@100;200;300;400;500;600;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap');
        
        /* Reset and Base Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .stApp {
            background: #000000;
            color: #ffffff;
            font-family: 'SF Pro Display', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 400;
            line-height: 1.6;
            overflow-x: hidden;
        }
        
        /* Apple-inspired Hero Section */
        .hero-section {
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            position: relative;
            background: radial-gradient(ellipse at center, #1a1a1a 0%, #000000 100%);
            text-align: center;
            margin: -2rem -2rem 0 -2rem;
            padding: 0 2rem;
        }
        
        .hero-title {
            font-size: clamp(3rem, 8vw, 7rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
            animation: fadeInUp 1s ease-out;
        }
        
        .hero-subtitle {
            font-size: clamp(1.2rem, 3vw, 1.8rem);
            font-weight: 300;
            color: #a0a0a0;
            max-width: 600px;
            margin: 0 auto 3rem auto;
            animation: fadeInUp 1s ease-out 0.2s both;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Tesla-inspired Floating Action Button */
        .floating-upload {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #ff3b30, #ff9500);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 25px rgba(255, 59, 48, 0.3);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            z-index: 1000;
        }
        
        .floating-upload:hover {
            transform: scale(1.1);
            box-shadow: 0 12px 35px rgba(255, 59, 48, 0.4);
        }
        
        /* Apple-inspired Content Sections */
        .content-section {
            min-height: 100vh;
            padding: 5rem 2rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
        }
        
        .section-title {
            font-size: clamp(2.5rem, 6vw, 4rem);
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }
        
        .section-subtitle {
            font-size: clamp(1rem, 2.5vw, 1.3rem);
            font-weight: 300;
            color: #a0a0a0;
            text-align: center;
            max-width: 600px;
            margin: 0 auto 4rem auto;
        }
        
        /* Tesla-inspired Grid System */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .feature-card {
            background: rgba(28, 28, 30, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2.5rem;
            text-align: center;
            backdrop-filter: blur(20px);
            transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            background: rgba(28, 28, 30, 0.9);
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        .feature-number {
            display: inline-block;
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #007AFF, #5856D6);
            border-radius: 50%;
            color: white;
            font-weight: 600;
            line-height: 40px;
            margin-bottom: 1.5rem;
            font-size: 1.1rem;
        }
        
        .feature-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #ffffff;
        }
        
        .feature-description {
            color: #a0a0a0;
            font-weight: 300;
            line-height: 1.5;
        }
        
        /* Revolutionary Upload Interface */
        .upload-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        .upload-zone {
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 24px;
            padding: 4rem 2rem;
            text-align: center;
            background: rgba(28, 28, 30, 0.6);
            backdrop-filter: blur(20px);
            transition: all 0.3s ease;
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
            border-color: rgba(0, 122, 255, 0.5);
            background: rgba(28, 28, 30, 0.8);
        }
        
        @keyframes rotate {
            100% { transform: rotate(360deg); }
        }
        
        .upload-icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 2rem auto;
            background: linear-gradient(135deg, #007AFF, #5856D6);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: white;
        }
        
        .upload-title {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .upload-subtitle {
            color: #a0a0a0;
            font-weight: 300;
            margin-bottom: 2rem;
        }
        
        /* Modern Action Button */
        .action-button {
            background: linear-gradient(135deg, #007AFF, #5856D6);
            border: none;
            border-radius: 12px;
            padding: 1rem 2.5rem;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            position: relative;
            overflow: hidden;
        }
        
        .action-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .action-button:hover::before {
            left: 100%;
        }
        
        .action-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 122, 255, 0.3);
        }
        
        /* Tesla-inspired Results Dashboard */
        .results-dashboard {
            background: rgba(28, 28, 30, 0.8);
            border-radius: 24px;
            padding: 3rem;
            margin: 3rem 0;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .dashboard-header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .dashboard-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .dashboard-subtitle {
            color: #a0a0a0;
            font-weight: 300;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }
        
        .metric-tile {
            background: rgba(44, 44, 46, 0.8);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .metric-tile:hover {
            background: rgba(44, 44, 46, 1);
            transform: translateY(-4px);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: #a0a0a0;
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9rem;
        }
        
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
        
        .result-info {
            flex: 1;
        }
        
        .result-filename {
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        
        .result-status {
            color: #a0a0a0;
            font-weight: 300;
        }
        
        .result-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
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
        
        /* Modern Theme Toggle */
        .theme-switcher {
            position: fixed;
            top: 2rem;
            right: 2rem;
            z-index: 1000;
            background: rgba(28, 28, 30, 0.8);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 50px;
            padding: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .theme-option {
            padding: 0.5rem 1rem;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        .theme-option.active {
            background: linear-gradient(135deg, #007AFF, #5856D6);
            color: white;
        }
        
        .theme-option:not(.active) {
            color: #a0a0a0;
        }
        
        .theme-option:not(.active):hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.1);
        }
        
        /* Smooth Scrolling */
        html {
            scroll-behavior: smooth;
        }
        
        /* Hide Streamlit Elements */
        .stApp > header {
            background-color: transparent;
        }
        
        .stApp > div:first-child {
            background-color: transparent;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        
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
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        </style>
        """
        
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border: 1px solid rgba(148, 163, 184, 0.1);
            padding: 2rem;
            border-radius: 16px;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
            margin: 1rem 0;
            color: #f1f5f9;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }
        
        .upload-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 2px dashed rgba(100, 116, 139, 0.3);
            color: #e2e8f0;
            padding: 3rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            position: relative;
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
        }
        
        .upload-box:hover {
            border-color: rgba(59, 130, 246, 0.5);
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        }
        
        .result-header {
            text-align: center;
            padding: 1rem 0;
            margin: 2rem 0 1rem 0;
            border-bottom: 2px solid #374151;
            color: #f9fafb;
        }
        
        .result-card {
            padding: 2rem;
            margin: 1.5rem 0;
            border-radius: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #f8fafc;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .result-card::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            border-radius: 50%;
            transform: translate(30px, -30px);
        }
        
        .result-card.fraud-high {
            background: linear-gradient(135deg, #7c2d12 0%, #991b1b 100%);
            box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2);
        }
        
        .result-card.fraud-medium {
            background: linear-gradient(135deg, #a16207 0%, #d97706 100%);
            box-shadow: 0 8px 32px rgba(245, 158, 11, 0.2);
        }
        
        .result-card.fraud-low {
            background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%);
            box-shadow: 0 8px 32px rgba(249, 115, 22, 0.2);
        }
        
        .result-card.legitimate {
            background: linear-gradient(135deg, #166534 0%, #15803d 100%);
            box-shadow: 0 8px 32px rgba(34, 197, 94, 0.2);
        }
        
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }
        
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 999;
            background: linear-gradient(135deg, #334155 0%, #475569 100%);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 12px;
            padding: 12px 20px;
            color: #f1f5f9;
            cursor: pointer;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .theme-toggle:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);
        }
        
        /* Enhanced dark mode text contrast */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #f8fafc !important;
        }
        
        .stMarkdown p {
            color: #e2e8f0 !important;
        }
        
        /* Button styling for dark mode */
        .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
        }
        
        /* File uploader styling for dark mode */
        .stFileUploader > div {
            background: #1f2937;
            border: 2px dashed #6b7280;
            border-radius: 10px;
        }
        
        .stFileUploader label {
            color: #f8fafc !important;
        }
        
        /* Expander styling for dark mode */
        .streamlit-expanderHeader {
            background: #1f2937;
            color: #f8fafc;
        }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Light Mode Theme */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            color: #1e293b;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
        }
        
        .main-header {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, #ffffff 0%, #3b82f6 50%, #ffffff 100%);
            color: #1e293b;
            border-radius: 0;
            margin: -2rem -2rem 3rem -2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(59, 130, 246, 0.05) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border: 1px solid rgba(59, 130, 246, 0.1);
            padding: 2rem;
            border-radius: 16px;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
            margin: 1rem 0;
            color: #0f172a;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
        }
        
        .upload-box {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border: 2px dashed rgba(59, 130, 246, 0.3);
            color: #334155;
            padding: 3rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            position: relative;
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
        }
        
        .upload-box:hover {
            border-color: rgba(59, 130, 246, 0.6);
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
        }
        
        .result-header {
            text-align: center;
            padding: 1rem 0;
            margin: 2rem 0 1rem 0;
            border-bottom: 2px solid #e5e7eb;
            color: #1f2937;
        }
        
        .result-card {
            padding: 2rem;
            margin: 1.5rem 0;
            border-radius: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.05);
            color: #0f172a;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .result-card::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
            border-radius: 50%;
            transform: translate(30px, -30px);
        }
        
        .result-card.fraud-high {
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
            box-shadow: 0 8px 32px rgba(239, 68, 68, 0.1);
        }
        
        .result-card.fraud-medium {
            background: linear-gradient(135deg, #fffbeb 0%, #fed7aa 100%);
            box-shadow: 0 8px 32px rgba(245, 158, 11, 0.1);
        }
        
        .result-card.fraud-low {
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            box-shadow: 0 8px 32px rgba(249, 115, 22, 0.1);
        }
        
        .result-card.legitimate {
            background: linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%);
            box-shadow: 0 8px 32px rgba(34, 197, 94, 0.1);
        }
        
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 999;
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            padding: 12px 20px;
            color: #0f172a;
            cursor: pointer;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }
        
        .theme-toggle:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12);
        }
        
        /* Enhanced light mode text contrast */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #1f2937 !important;
        }
        
        .stMarkdown p {
            color: #374151 !important;
        }
        
        /* Button styling for light mode */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }
        
        /* File uploader styling for light mode */
        .stFileUploader > div {
            background: #f8fafc;
            border: 2px dashed #cbd5e1;
            border-radius: 10px;
        }
        
        .stFileUploader label {
            color: #1f2937 !important;
        }
        
        /* Expander styling for light mode */
        .streamlit-expanderHeader {
            background: #f8fafc;
            color: #1f2937;
        }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# --- Theme Toggle Button ---
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    theme_text = "Dark Mode" if not st.session_state.dark_mode else "Light Mode"
    
    if st.button(theme_text, key="theme_toggle", help="Switch between light and dark themes"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.experimental_rerun()

# --- Main Header ---
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3rem; font-weight: 700; margin: 0 0 1rem 0; background: linear-gradient(135deg, #1e293b, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Vehicle Damage Fraud Detector</h1>
    <p style="font-size: 1.3em; margin: 0; opacity: 0.9; font-weight: 400;">Advanced AI-powered fraud detection for vehicle damage claims</p>
</div>
""", unsafe_allow_html=True)

# --- How It Works Section ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #3b82f6; font-weight: 600;">Step 1: Upload</h3>
        <p style="opacity: 0.8;">Upload one or more vehicle damage photos</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #8b5cf6; font-weight: 600;">Step 2: Analyze</h3>
        <p style="opacity: 0.8;">AI analyzes images for fraud indicators</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #06b6d4; font-weight: 600;">Step 3: Results</h3>
        <p style="opacity: 0.8;">Get instant fraud risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Upload Section ---
st.markdown("""
<div class="upload-box">
    <h3 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem;">Upload Vehicle Damage Images</h3>
    <p style="opacity: 0.8; margin: 0;">Supported formats: JPG, JPEG, PNG • Multiple files allowed</p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload clear photos of vehicle damage for best results"
)

# --- Model Loading ---
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

# Load performance summary
@st.cache_data
def load_performance_summary():
    """Load model performance metrics"""
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(project_root, "models", "artifacts", "model_performance_summary.txt")
    
    try:
        with open(summary_path, 'r') as f:
            return f.read()
    except:
        return "Performance summary not available."

# --- Real-time Threshold Monitor ---
with st.sidebar.expander("🎯 Current Threshold", expanded=True):
    threshold, mod_time = get_current_threshold()
    
    # Initialize session state for threshold tracking
    if 'last_threshold' not in st.session_state:
        st.session_state.last_threshold = threshold
    if 'threshold_updated' not in st.session_state:
        st.session_state.threshold_updated = False
    
    # Check if threshold changed
    if st.session_state.last_threshold != threshold:
        st.session_state.threshold_updated = True
        st.session_state.last_threshold = threshold
    else:
        st.session_state.threshold_updated = False
    
    # Create columns for threshold display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show threshold with change indicator
        delta = None
        if st.session_state.threshold_updated:
            delta = "Updated!"
            
        st.metric(
            label="Active Threshold", 
            value=f"{threshold:.5f}",
            delta=delta,
            help=f"Live threshold - updates automatically when file changes"
        )
    
    with col2:
        # Manual refresh button
        if st.button("🔄", help="Refresh now"):
            st.experimental_rerun()
    
    # Show update status
    if st.session_state.threshold_updated:
        st.success("🎯 Threshold updated! New predictions will use this value.")
    
    st.caption(f"📅 File modified: {mod_time}")
    
    # Auto-refresh options
    auto_refresh = st.selectbox(
        "Auto-refresh", 
        ["Manual", "Every 5 seconds", "Every 10 seconds", "Every 30 seconds"],
        index=0,
        help="Automatically check for threshold file changes"
    )
    
    # Implement auto-refresh
    if auto_refresh != "Manual":
        refresh_seconds = {"Every 5 seconds": 5, "Every 10 seconds": 10, "Every 30 seconds": 30}[auto_refresh]
        st.info(f"⏱️ Auto-refreshing every {refresh_seconds} seconds...")
        time.sleep(refresh_seconds)
        st.experimental_rerun()

# --- Threshold Editor ---
with st.sidebar.expander("⚙️ Threshold Editor", expanded=False):
    st.write("**Change threshold in real-time:**")
    
    # Get current threshold for the editor
    current_threshold, _ = get_current_threshold()
    
    # Threshold input
    new_threshold = st.number_input(
        "New Threshold Value",
        min_value=0.0,
        max_value=1.0,
        value=float(current_threshold),
        step=0.001,
        format="%.5f",
        help="Higher = fewer fraud predictions, Lower = more fraud predictions"
    )
    
    # Preset buttons
    st.write("**Quick Presets:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Conservative\n0.69770", help="Minimize false alarms"):
            new_threshold = 0.69770
    
    with col2:
        if st.button("Balanced\n0.59650", help="Original optimized"):
            new_threshold = 0.59650
    
    with col3:
        if st.button("Sensitive\n0.42000", help="Catch more fraud"):
            new_threshold = 0.42000
    
    # Update threshold button
    if st.button("💾 Update Threshold", type="primary"):
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            threshold_path = os.path.join(project_root, "models", "production", "optimal_threshold.txt")
            
            with open(threshold_path, 'w') as f:
                f.write(str(new_threshold))
            
            st.success(f"✅ Threshold updated to {new_threshold:.5f}")
            st.info("🔄 Predictions will now use the new threshold!")
            time.sleep(1)
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to update threshold: {str(e)}")
    
    # Impact preview
    if abs(new_threshold - current_threshold) > 0.001:
        st.write("**Impact Preview:**")
        if new_threshold > current_threshold:
            st.info("📉 Higher threshold → Fewer fraud predictions → More conservative")
        else:
            st.info("📈 Lower threshold → More fraud predictions → More sensitive")

# --- File Upload Status ---
if not uploaded_files:
    st.info("👆 **Please upload one or more images above to start fraud detection analysis**")
    st.write("**Supported formats:** JPG, JPEG, PNG")
    st.write("**Multiple files:** You can upload multiple images at once for batch processing")

if uploaded_files:
    st.success(f"{len(uploaded_files)} image(s) uploaded.")
    st.write("Preview:")
    cols = st.columns(min(4, len(uploaded_files)))
    for idx, file in enumerate(uploaded_files):
        file.seek(0)  # Reset file pointer
        img = Image.open(file)
        cols[idx % 4].image(img, caption=file.name, use_column_width=True)
    
    # Show model performance summary
    with st.expander("📊 Model Performance Summary", expanded=False):
        summary = load_performance_summary()
        st.text(summary)
    
    # Model test with dummy data
    if st.button("🧪 Test Optimized Model"):
        model, threshold, mod_time = load_optimized_model()
        
        if model is None:
            st.error("❌ Model could not be loaded. Please check the model files.")
            st.stop()
            
        st.info(f"Using optimized threshold: {threshold:.5f} (updated: {mod_time})")
        
        # Test 1: Random data
        test_data = np.random.rand(1, 256, 256, 3).astype('float32')
        pred_prob = float(model.predict(test_data, verbose=0)[0][0])
        pred_class = "Fraud" if pred_prob >= threshold else "Non-Fraud"
        st.write(f"Random data - Probability: {pred_prob:.4f}, Prediction: {pred_class}")
        
        # Test 2: All zeros (black image)
        test_zeros = np.zeros((1, 256, 256, 3), dtype='float32')
        pred_prob_zeros = float(model.predict(test_zeros, verbose=0)[0][0])
        pred_class_zeros = "Fraud" if pred_prob_zeros >= threshold else "Non-Fraud"
        st.write(f"Black image - Probability: {pred_prob_zeros:.4f}, Prediction: {pred_class_zeros}")
        
        # Test 3: All ones (white image)
        test_ones = np.ones((1, 256, 256, 3), dtype='float32')
        pred_prob_ones = float(model.predict(test_ones, verbose=0)[0][0])
        pred_class_ones = "Fraud" if pred_prob_ones >= threshold else "Non-Fraud"
        st.write(f"White image - Probability: {pred_prob_ones:.4f}, Prediction: {pred_class_ones}")
        
        # Test variance
        variance = max(pred_prob, pred_prob_zeros, pred_prob_ones) - min(pred_prob, pred_prob_zeros, pred_prob_ones)
        if variance > 0.1:
            st.success(f"✅ Model shows good variance ({variance:.3f}) - Working properly!")
        else:
            st.warning(f"⚠️ Low variance ({variance:.3f}) - Model may need improvement")

    # --- Analyze Button ---
    if st.button("� Analyze Images for Fraud", type="primary", use_container_width=True):
        with st.spinner("🤖 AI is analyzing your images..."):
            model, threshold, mod_time = load_optimized_model()
            
            if model is None:
                st.error("❌ Unable to load the AI model. Please try again.")
                st.stop()
                
            results = []
            st.info(f"🎯 Using AI threshold: {threshold:.3f}")
            
            for file in uploaded_files:
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
                prediction = "Fraud" if is_fraud else "Non-Fraud"
                
                # Calculate confidence
                if is_fraud:
                    confidence = pred_probability
                else:
                    confidence = 1.0 - pred_probability
                
                # Generate evidence
                if is_fraud:
                    if pred_probability > 0.8:
                        evidence = f"🔴 HIGH FRAUD RISK (Probability: {pred_probability:.3f}) - Requires immediate investigation"
                    elif pred_probability > 0.6:
                        evidence = f"🟡 MODERATE FRAUD RISK (Probability: {pred_probability:.3f}) - Manual review recommended"
                    else:
                        evidence = f"🟠 LOW FRAUD RISK (Probability: {pred_probability:.3f}) - Above threshold but borderline"
                else:
                    if pred_probability < 0.2:
                        evidence = f"✅ VERY LOW FRAUD RISK (Probability: {pred_probability:.3f}) - Likely legitimate"
                    else:
                        evidence = f"✅ LOW FRAUD RISK (Probability: {pred_probability:.3f}) - Below threshold"
                
                results.append({
                    "Image": file.name,
                    "Prediction": prediction,
                    "Probability": f"{pred_probability:.4f}",
                    "Confidence": f"{confidence:.3f}",
                    "Risk Level": "HIGH" if pred_probability > 0.8 else "MODERATE" if pred_probability > 0.6 else "LOW",
                    "Evidence": evidence
                })
            
            # Show results with better formatting
            st.markdown("""
            <div class="result-header">
                <h2>📊 Analysis Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary statistics with better visuals
            fraud_count = len([r for r in results if r["Prediction"] == "Fraud"])
            total_count = len(results)
            legitimate_count = total_count - fraud_count
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 Total Images</h3>
                    <h2 style="color: #1f77b4;">{total_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fraud_color = "#d32f2f" if fraud_count > 0 else "#4caf50"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚠️ Fraud Detected</h3>
                    <h2 style="color: {fraud_color};">{fraud_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>✅ Legitimate</h3>
                    <h2 style="color: #4caf50;">{legitimate_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Create results dataframe for detailed view
            df = pd.DataFrame(results)
            
            # Individual results with cards
            st.markdown("<h3>🔍 Detailed Analysis</h3>", unsafe_allow_html=True)
            
            for i, result in enumerate(results):
                is_fraud = result["Prediction"] == "Fraud"
                prob = float(result["Probability"])
                
                # Determine card styling based on result
                if is_fraud:
                    if prob > 0.8:
                        card_class = "fraud-high"
                        icon = "🚨"
                        status = "HIGH RISK"
                    elif prob > 0.6:
                        card_class = "fraud-medium"
                        icon = "⚠️"
                        status = "MODERATE RISK"
                    else:
                        card_class = "fraud-low"
                        icon = "🟡"
                        status = "LOW RISK"
                else:
                    card_class = "legitimate"
                    icon = "✅"
                    status = "LEGITIMATE"
                
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4>{icon} {result['Image']}</h4>
                            <p><strong>Status:</strong> {status}</p>
                            <p><strong>Confidence:</strong> {result['Confidence']}</p>
                        </div>
                        <div style="text-align: right;">
                            <h3>{prob:.1%}</h3>
                            <small>Fraud Probability</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Option to show detailed table
            with st.expander("📋 View Detailed Table"):
                # Color-code the results
                def color_prediction(val):
                    if val == "Fraud":
                        return 'background-color: #ffebee; color: #c62828'
                    else:
                        return 'background-color: #e8f5e8; color: #2e7d32'
                
                def color_risk(val):
                    if val == "HIGH":
                        return 'background-color: #ffcdd2; font-weight: bold'
                    elif val == "MODERATE":
                        return 'background-color: #fff3e0'
                    else:
                        return 'background-color: #f1f8e9'
                
                styled_df = df.style.applymap(color_prediction, subset=['Prediction'])
                styled_df = styled_df.applymap(color_risk, subset=['Risk Level'])
                
                st.dataframe(styled_df, use_container_width=True)

            # --- Export Options ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv,
                    file_name=f"fraud_analysis_{len(results)}_images.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Create a summary report
                summary_text = f"""Vehicle Damage Fraud Analysis Report
                
Total Images Analyzed: {total_count}
Fraudulent Claims Detected: {fraud_count}
Legitimate Claims: {legitimate_count}
Analysis Threshold: {threshold:.3f}

Detailed Results:
{df.to_string(index=False)}
"""
                st.download_button(
                    label="📄 Download Report",
                    data=summary_text.encode("utf-8"),
                    file_name=f"fraud_analysis_report_{len(results)}_images.txt",
                    mime="text/plain",
                    use_container_width=True
            )

            # --- Evidence Section ---
            st.subheader("Evidence & Reasoning")
            for r in results:
                # Color-code based on prediction
                color = "🔴" if r['Prediction'] == "Fraud" else "🟢"
                
                st.markdown(
                    f"{color} **{r['Image']}**: {r['Prediction']} (Confidence: {r['Confidence']})\n\n"
                    f"📊 Evidence: {r['Evidence']}\n\n"
                    f"💡 Reasoning: EfficientNet model analyzed image features. "
                    f"Values < 0.5 indicate potential fraud, > 0.5 indicate legitimate claims. "
                    f"Higher confidence means the prediction is further from the 0.5 threshold."
                )

else:
    # Help section when no images uploaded
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Getting Started</h3>
        <p>📸 <strong>Upload Images:</strong> Select clear photos of vehicle damage</p>
        <p>🤖 <strong>AI Analysis:</strong> Our system will analyze each image for fraud indicators</p>
        <p>📊 <strong>Get Results:</strong> View detailed fraud risk assessment with confidence scores</p>
        <p>💾 <strong>Export Data:</strong> Download results as CSV or summary report</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-box">
        <h3>Ready to Start?</h3>
        <p>Upload your vehicle damage images above to begin the analysis</p>
    </div>
    """, unsafe_allow_html=True)

# --- Help & FAQ Section ---
with st.expander("❓ Help & FAQ"):
    st.markdown("""
    **How accurate is the fraud detection?**
    - Our AI model has been optimized to minimize false alarms while maintaining high detection rates
    - Current threshold is set to ~60% to balance sensitivity and specificity
    
    **What image formats are supported?**
    - JPG, JPEG, and PNG formats are accepted
    - Images are automatically resized for optimal processing
    
    **How should I interpret the results?**
    - 🚨 **High Risk (>80%):** Requires immediate investigation
    - ⚠️ **Moderate Risk (60-80%):** Manual review recommended  
    - 🟡 **Low Risk (threshold-80%):** Borderline cases
    - ✅ **Legitimate (<threshold):** Likely authentic claims
    
    **Can I process multiple images at once?**
    - Yes! Upload multiple images and get batch analysis results
    - Results can be exported as CSV or summary report
    """)

# --- Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; border-top: 1px solid #e0e0e0;">
    <p>🚗 <strong>Vehicle Damage Fraud Detector</strong> | Powered by AI & Machine Learning</p>
    <p><small>Built with EfficientNetV2-B0 • Optimized for Insurance Claims • Version 2.0</small></p>
</div>
""", unsafe_allow_html=True)