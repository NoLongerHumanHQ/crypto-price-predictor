import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import numpy as np

def setup_page_config():
    """
    Configure the Streamlit page settings
    """
    st.set_page_config(
        page_title="Crypto Price Predictor",
        page_icon="🪙",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/crypto-price-predictor',
            'Report a bug': 'https://github.com/yourusername/crypto-price-predictor/issues',
            'About': "# Crypto Price Predictor\nA lightweight cryptocurrency price prediction app."
        }
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 10px;
    }
    .stSelectbox label, .stSlider label {
        font-weight: bold;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

def format_large_number(num):
    """
    Format large numbers in a readable format
    
    Args:
        num (float): Number to format
        
    Returns:
        str: Formatted number string
    """
    if num is None:
        return "N/A"
        
    if num >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"${num / 1_000:.2f}K"
    else:
        return f"${num:.2f}"

def format_percent(value):
    """
    Format percentage values
    
    Args:
        value (float): Percentage value
        
    Returns:
        str: Formatted percentage string
    """
    if value is None:
        return "N/A"
        
    return f"{value:.2f}%" if value > 0 else f"{value:.2f}%"

def get_crypto_icon(crypto_id, fallback_icon=None):
    """
    Get the cryptocurrency logo/icon from CoinGecko
    
    Args:
        crypto_id (str): Cryptocurrency ID
        fallback_icon (str, optional): Fallback icon URL
        
    Returns:
        str: URL of the cryptocurrency icon
    """
    # Some popular crypto icons (to avoid API rate limits)
    icon_mapping = {
        'bitcoin': 'https://assets.coingecko.com/coins/images/1/small/bitcoin.png',
        'ethereum': 'https://assets.coingecko.com/coins/images/279/small/ethereum.png',
        'tether': 'https://assets.coingecko.com/coins/images/325/small/Tether.png',
        'binancecoin': 'https://assets.coingecko.com/coins/images/825/small/bnb-icon2_2x.png',
        'solana': 'https://assets.coingecko.com/coins/images/4128/small/solana.png',
        'xrp': 'https://assets.coingecko.com/coins/images/44/small/xrp-symbol-white-128.png',
        'cardano': 'https://assets.coingecko.com/coins/images/975/small/cardano.png',
        'dogecoin': 'https://assets.coingecko.com/coins/images/5/small/dogecoin.png',
        'polkadot': 'https://assets.coingecko.com/coins/images/12171/small/aJGBjJFU_400x400.jpg',
        'tron': 'https://assets.coingecko.com/coins/images/1094/small/tron-logo.png',
    }
    
    # If we have a cached icon, return it
    if crypto_id.lower() in icon_mapping:
        return icon_mapping[crypto_id.lower()]
    
    # If not in our mapping, try to fetch from CoinGecko's API
    try:
        # We'll use a simple static icon to avoid API calls
        return fallback_icon or "https://cryptologos.cc/logos/generic-placeholder-logo.png"
    except Exception:
        # If all else fails, return a generic icon
        return "https://cryptologos.cc/logos/generic-placeholder-logo.png"

def load_model_presets():
    """
    Load model presets/hyperparameters
    
    Returns:
        dict: Dictionary with model presets
    """
    presets = {
        "Linear Regression": {
            "alpha": 1.0,
            "fit_intercept": True,
            "window_size": 7
        },
        "ARIMA": {
            "p": 2,
            "d": 1,
            "q": 2
        },
        "Random Forest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "window_size": 7
        },
        "SVR": {
            "kernel": "rbf",
            "C": 100,
            "gamma": 0.1,
            "epsilon": 0.1,
            "window_size": 5
        },
        "Moving Average": {
            "window_size": 7
        },
        "Polynomial Regression": {
            "degree": 2,
            "alpha": 1.0,
            "window_size": 5
        }
    }
    
    return presets 