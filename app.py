import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data_handler import CryptoDataHandler
from model import PredictionEngine
from utils import setup_page_config, get_crypto_icon, format_large_number, format_percent

def main():
    # Setup page configuration
    setup_page_config()
    
    # Initialize data handler
    data_handler = CryptoDataHandler()
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Cryptocurrency selector
        available_cryptos = data_handler.get_available_cryptocurrencies()
        selected_crypto = st.selectbox(
            "Select Cryptocurrency",
            options=available_cryptos,
            index=0,
            format_func=lambda x: f"{data_handler.get_crypto_name(x)} ({x.upper()})"
        )
        
        # Time period selector
        time_periods = {
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "1y": 365
        }
        selected_period = st.select_slider(
            "Historical Time Period",
            options=list(time_periods.keys()),
            value="30d"
        )
        days = time_periods[selected_period]
        
        # Prediction horizon selector
        prediction_horizons = {
            "1d": 1,
            "7d": 7,
            "30d": 30
        }
        prediction_horizon = st.select_slider(
            "Prediction Horizon",
            options=list(prediction_horizons.keys()),
            value="7d"
        )
        forecast_days = prediction_horizons[prediction_horizon]
        
        # Model selection
        available_models = [
            "Linear Regression",
            "ARIMA",
            "Random Forest",
            "SVR",
            "Moving Average",
            "Polynomial Regression"
        ]
        selected_model = st.radio("Prediction Model", available_models)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            show_indicators = st.checkbox("Show Technical Indicators", value=False)
            show_confidence = st.checkbox("Show Confidence Intervals", value=True)
            
            if selected_model == "Linear Regression":
                st.slider("Regularization Strength", 0.01, 10.0, 1.0, 0.01)
            elif selected_model == "ARIMA":
                st.number_input("AR(p)", 1, 5, 2)
                st.number_input("I(d)", 0, 2, 1)
                st.number_input("MA(q)", 0, 5, 2)
            elif selected_model == "Random Forest":
                st.slider("Number of Trees", 10, 200, 100, 10)
            elif selected_model == "SVR":
                st.selectbox("Kernel", ["linear", "poly", "rbf"])
            elif selected_model == "Moving Average":
                st.slider("Window Size", 3, 30, 7)
            elif selected_model == "Polynomial Regression":
                st.slider("Degree", 2, 6, 2)
    
    # Main content
    st.title("🪙 Crypto Price Predictor")
    
    # Load data
    with st.spinner(f"Loading {selected_crypto.upper()} data..."):
        df = data_handler.get_historical_data(selected_crypto, days)
        
    if df is None or len(df) == 0:
        st.error(f"Failed to load data for {selected_crypto.upper()}. Please try another cryptocurrency or check your internet connection.")
        return
    
    # Initialize prediction engine
    prediction_engine = PredictionEngine(selected_model)
    
    # Current price and info display
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    current_price = df['close'].iloc[-1]
    day_change = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
    
    with col1:
        icon_url = get_crypto_icon(selected_crypto)
        st.image(icon_url, width=50)
        st.markdown(f"### {selected_crypto.upper()}")
        
    with col2:
        st.metric(
            "Current Price", 
            f"${current_price:.2f}", 
            f"{day_change:.2f}%" if day_change > 0 else f"{day_change:.2f}%",
            delta_color="normal"
        )
        
    with col3:
        market_cap = data_handler.get_market_cap(selected_crypto)
        st.metric("Market Cap", format_large_number(market_cap))
    
    with col4:
        volume = data_handler.get_24h_volume(selected_crypto)
        st.metric("24h Volume", format_large_number(volume))
    
    # Price Chart
    st.subheader("Historical Price")
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        )
    )
    
    # Show technical indicators if enabled
    if show_indicators:
        # Add 20-day Moving Average
        ma20 = df['close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(color='blue', width=1), name='20-day MA'))
        
        # Add 50-day Moving Average
        ma50 = df['close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma50, line=dict(color='orange', width=1), name='50-day MA'))
    
    fig.update_layout(
        title=f"{selected_crypto.upper()} Price Chart ({selected_period})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Price predictions
    st.subheader(f"Price Prediction ({prediction_horizon})")
    
    with st.spinner("Generating predictions..."):
        # Prepare features from historical data
        prediction_data = prediction_engine.prepare_features(df)
        
        # Train model and make predictions
        predictions, confidence_intervals, accuracy = prediction_engine.predict(
            prediction_data, 
            forecast_days
        )
        
        # Format the prediction dates
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predictions
        })
        
        if show_confidence and confidence_intervals is not None:
            pred_df['lower_bound'] = confidence_intervals[:, 0]
            pred_df['upper_bound'] = confidence_intervals[:, 1]
    
    # Display predicted change
    predicted_change_pct = ((predictions[-1] / current_price) - 1) * 100
    
    st.metric(
        f"Predicted Price ({future_dates[-1].strftime('%Y-%m-%d')})", 
        f"${predictions[-1]:.2f}", 
        f"{predicted_change_pct:.2f}%",
        delta_color="normal"
    )
    
    # Plot predictions
    fig2 = go.Figure()
    
    # Add historical prices
    fig2.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        )
    )
    
    # Add predictions
    fig2.add_trace(
        go.Scatter(
            x=pred_df['date'],
            y=pred_df['predicted_price'],
            mode='lines',
            name='Predicted Price',
            line=dict(color='green', dash='dash')
        )
    )
    
    # Add confidence intervals if enabled
    if show_confidence and 'lower_bound' in pred_df and 'upper_bound' in pred_df:
        fig2.add_trace(
            go.Scatter(
                x=pred_df['date'].tolist() + pred_df['date'].tolist()[::-1],
                y=pred_df['upper_bound'].tolist() + pred_df['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,176,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            )
        )
    
    fig2.update_layout(
        title=f"{selected_crypto.upper()} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Model performance metrics
    st.subheader("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy['accuracy']:.2f}%")
    
    with col2:
        st.metric("RMSE", f"{accuracy['rmse']:.4f}")
    
    with col3:
        st.metric("MAE", f"{accuracy['mae']:.4f}")
    
    # Display model details
    with st.expander("Model Details"):
        st.markdown(f"""
        **Model Type:** {selected_model}
        
        **Training Data:** {len(df)} days of historical data
        
        **Features Used:**
        - Price (Open, High, Low, Close)
        - Volume
        - Moving Averages
        - Volatility
        """)
        
        if selected_model == "Ensemble":
            st.markdown("Using ensemble of multiple models for improved prediction accuracy")

if __name__ == "__main__":
    main() 