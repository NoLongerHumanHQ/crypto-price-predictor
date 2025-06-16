# Crypto Price Predictor

A lightweight cryptocurrency price prediction web application built with Streamlit that provides price forecasts using various machine learning models.

## Features

- Real-time cryptocurrency data from CoinGecko API
- Multiple prediction models:
  - Linear Regression
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Random Forest
  - SVR (Support Vector Regression)
  - Moving Average
  - Polynomial Regression
- Interactive visualization of historical prices and forecasts
- Technical indicators (RSI, MACD, Bollinger Bands)
- Customizable time periods and prediction horizons
- Mobile-responsive design

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crypto-price-predictor.git
   cd crypto-price-predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

The app will be accessible at `http://localhost:8501` in your web browser.

## Application Structure

- `app.py`: Main Streamlit application entry point
- `data_handler.py`: Handles API calls and data processing
- `model.py`: Contains prediction logic and ML models
- `utils.py`: Helper functions and configurations
- `requirements.txt`: All necessary dependencies

## API Usage Notes

This application uses the free tier of the CoinGecko API, which has rate limits. The app implements rate limiting and caching to respect these limits. If you encounter API errors, please wait a few minutes before trying again.

## Models

### Linear Regression
A simple baseline model that predicts future prices based on linear relationships between features.

### ARIMA
Time series forecasting model that captures autocorrelation in the price data.

### Random Forest
Ensemble learning method that fits multiple decision trees to the data to make more robust predictions.

### SVR (Support Vector Regression)
A non-linear regression approach that uses support vector machines to predict prices.

### Moving Average
A simple technical analysis approach that uses the average of historical prices to predict future prices.

### Polynomial Regression
Extends linear regression to capture non-linear relationships in the price data.

## License

MIT

## Disclaimer

This application is for educational and informational purposes only. Cryptocurrency investments are volatile and risky. Do not use predictions from this application as financial advice. Always conduct your own research before making investment decisions. 
