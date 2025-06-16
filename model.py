import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

class PredictionEngine:
    """
    Prediction engine for cryptocurrency price prediction.
    Supports multiple prediction models.
    """
    
    def __init__(self, model_name):
        """
        Initialize the prediction engine with the selected model
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.trained = False
    
    def prepare_features(self, df):
        """
        Prepare features for the model
        
        Args:
            df (pd.DataFrame): DataFrame with historical price data
            
        Returns:
            dict: Dictionary with processed features
        """
        # Make a copy of the dataframe
        data = df.copy()
        
        # Extract features
        features = {
            'close': data['close'].values,
            'open': data['open'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            'volume': data['volume'].values,
            'dates': data.index.values,
            'sma_5': data['sma_5'].values if 'sma_5' in data.columns else None,
            'sma_20': data['sma_20'].values if 'sma_20' in data.columns else None,
            'ema_12': data['ema_12'].values if 'ema_12' in data.columns else None,
            'ema_26': data['ema_26'].values if 'ema_26' in data.columns else None,
            'macd': data['macd'].values if 'macd' in data.columns else None,
            'rsi': data['rsi'].values if 'rsi' in data.columns else None,
            'bb_upper': data['bb_upper'].values if 'bb_upper' in data.columns else None,
            'bb_lower': data['bb_lower'].values if 'bb_lower' in data.columns else None,
            'volatility': data['volatility'].values if 'volatility' in data.columns else None
        }
        
        # Add lagged features
        for lag in [1, 2, 3, 5, 7]:
            features[f'close_lag_{lag}'] = np.roll(data['close'].values, lag)
            
        # Add percentage changes
        features['pct_change_1d'] = np.concatenate([[0], np.diff(data['close'].values) / data['close'].values[:-1]])
        
        # Replace NaN and infinite values
        for key, value in features.items():
            if value is not None and key != 'dates':
                features[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def predict(self, features, forecast_days):
        """
        Train the model and make predictions
        
        Args:
            features (dict): Dictionary with features
            forecast_days (int): Number of days to forecast
            
        Returns:
            tuple: Predictions, confidence intervals, accuracy metrics
        """
        if self.model_name == "Linear Regression":
            return self._predict_linear_regression(features, forecast_days)
        elif self.model_name == "ARIMA":
            return self._predict_arima(features, forecast_days)
        elif self.model_name == "Random Forest":
            return self._predict_random_forest(features, forecast_days)
        elif self.model_name == "SVR":
            return self._predict_svr(features, forecast_days)
        elif self.model_name == "Moving Average":
            return self._predict_moving_average(features, forecast_days)
        elif self.model_name == "Polynomial Regression":
            return self._predict_polynomial_regression(features, forecast_days)
        else:
            # Default to linear regression
            return self._predict_linear_regression(features, forecast_days)
    
    def _prepare_train_test_data(self, features, window_size=7, test_size=0.2):
        """
        Prepare training and testing data
        
        Args:
            features (dict): Dictionary with features
            window_size (int): Size of the window for sequence prediction
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        close_prices = features['close']
        
        # Determine split point
        split_idx = int(len(close_prices) * (1 - test_size))
        
        # Create sequences
        X = []
        y = []
        
        for i in range(window_size, len(close_prices)):
            X.append(close_prices[i - window_size:i])
            y.append(close_prices[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split the data
        X_train = X[:split_idx - window_size]
        X_test = X[split_idx - window_size:]
        y_train = y[:split_idx - window_size]
        y_test = y[split_idx - window_size:]
        
        return X_train, X_test, y_train, y_test
    
    def _predict_linear_regression(self, features, forecast_days):
        """
        Make predictions using Linear Regression
        
        Args:
            features (dict): Dictionary with features
            forecast_days (int): Number of days to forecast
            
        Returns:
            tuple: Predictions, confidence intervals, accuracy metrics
        """
        # Prepare sequence data
        X_train, X_test, y_train, y_test = self._prepare_train_test_data(features)
        
        # Reshape for Linear Regression
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_flat, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_flat)
        accuracy = self._calculate_accuracy(y_test, y_pred)
        
        # Make future predictions
        last_window = features['close'][-7:]
        predictions = []
        confidence = []
        
        for _ in range(forecast_days):
            input_data = last_window.reshape(1, -1)
            prediction = model.predict(input_data)[0]
            
            # Simple confidence interval calculation
            std = np.std(y_test - y_pred)
            lower_bound = prediction - 1.96 * std
            upper_bound = prediction + 1.96 * std
            
            predictions.append(prediction)
            confidence.append([lower_bound, upper_bound])
            
            # Update window
            last_window = np.append(last_window[1:], prediction)
        
        return np.array(predictions), np.array(confidence), accuracy
    
    def _predict_arima(self, features, forecast_days):
        """
        Make predictions using ARIMA
        
        Args:
            features (dict): Dictionary with features
            forecast_days (int): Number of days to forecast
            
        Returns:
            tuple: Predictions, confidence intervals, accuracy metrics
        """
        # Use only the close prices
        close_prices = features['close']
        
        # Split data
        split_idx = int(0.8 * len(close_prices))
        train, test = close_prices[:split_idx], close_prices[split_idx:]
        
        # Fit ARIMA model
        try:
            model = ARIMA(train, order=(2, 1, 2))
            model_fit = model.fit()
            
            # Make predictions on test set
            y_pred = model_fit.forecast(steps=len(test))
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(test, y_pred)
            
            # Make future predictions
            forecast_result = model_fit.get_forecast(steps=forecast_days)
            predictions = forecast_result.predicted_mean
            
            # Get confidence intervals
            conf_int = forecast_result.conf_int(alpha=0.05)
            confidence = np.array([conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values]).T
        
        except Exception as e:
            # If ARIMA fails, fall back to moving average
            st.warning(f"ARIMA model failed: {e}. Falling back to moving average.")
            return self._predict_moving_average(features, forecast_days)
        
        return predictions, confidence, accuracy
    
    def _predict_random_forest(self, features, forecast_days):
        """
        Make predictions using Random Forest
        
        Args:
            features (dict): Dictionary with features
            forecast_days (int): Number of days to forecast
            
        Returns:
            tuple: Predictions, confidence intervals, accuracy metrics
        """
        # Prepare sequence data
        X_train, X_test, y_train, y_test = self._prepare_train_test_data(features)
        
        # Reshape for Random Forest
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_flat, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_flat)
        accuracy = self._calculate_accuracy(y_test, y_pred)
        
        # Make future predictions
        last_window = features['close'][-7:]
        predictions = []
        confidence = []
        
        for _ in range(forecast_days):
            input_data = last_window.reshape(1, -1)
            prediction = model.predict(input_data)[0]
            
            # Get prediction intervals from Random Forest
            pred_intervals = np.zeros((100, 1))
            for i, tree in enumerate(model.estimators_):
                pred_intervals[i] = tree.predict(input_data)
            
            lower_bound = np.percentile(pred_intervals, 2.5)
            upper_bound = np.percentile(pred_intervals, 97.5)
            
            predictions.append(prediction)
            confidence.append([lower_bound, upper_bound])
            
            # Update window
            last_window = np.append(last_window[1:], prediction)
        
        return np.array(predictions), np.array(confidence), accuracy
    
    def _predict_svr(self, features, forecast_days):
        """
        Make predictions using SVR
        
        Args:
            features (dict): Dictionary with features
            forecast_days (int): Number of days to forecast
            
        Returns:
            tuple: Predictions, confidence intervals, accuracy metrics
        """
        # Prepare sequence data
        X_train, X_test, y_train, y_test = self._prepare_train_test_data(features, window_size=5)
        
        # Reshape for SVR
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train model
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        model.fit(X_train_flat, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_flat)
        accuracy = self._calculate_accuracy(y_test, y_pred)
        
        # Make future predictions
        last_window = features['close'][-5:]
        predictions = []
        
        for _ in range(forecast_days):
            input_data = last_window.reshape(1, -1)
            prediction = model.predict(input_data)[0]
            predictions.append(prediction)
            
            # Update window
            last_window = np.append(last_window[1:], prediction)
        
        # SVR doesn't provide confidence intervals directly
        # Use a simple approach based on the mean error
        mean_error = np.mean(np.abs(y_test - y_pred))
        confidence = np.array([[p - 1.96 * mean_error, p + 1.96 * mean_error] for p in predictions])
        
        return np.array(predictions), confidence, accuracy
    
    def _predict_moving_average(self, features, forecast_days):
        """
        Make predictions using Moving Average
        
        Args:
            features (dict): Dictionary with features
            forecast_days (int): Number of days to forecast
            
        Returns:
            tuple: Predictions, confidence intervals, accuracy metrics
        """
        # Use only the close prices
        close_prices = features['close']
        
        # Configuration
        window_size = 7
        
        # Split data for evaluation
        split_idx = int(0.8 * len(close_prices))
        train, test = close_prices[:split_idx], close_prices[split_idx:]
        
        # Calculate moving average on test data
        y_pred = np.array([np.mean(train[-window_size:]) for _ in range(len(test))])
        
        # For each test point, update the window and calculate the next prediction
        for i in range(len(test) - 1):
            window = np.append(train[-window_size+1:], test[:i+1])
            y_pred[i+1] = np.mean(window[-window_size:])
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(test, y_pred)
        
        # Make future predictions
        predictions = []
        confidence = []
        window = close_prices[-window_size:]
        
        # Historical volatility for confidence intervals
        volatility = np.std(np.diff(close_prices[-30:])) if len(close_prices) >= 30 else np.std(np.diff(close_prices))
        
        for _ in range(forecast_days):
            prediction = np.mean(window)
            lower_bound = prediction - 1.96 * volatility
            upper_bound = prediction + 1.96 * volatility
            
            predictions.append(prediction)
            confidence.append([lower_bound, upper_bound])
            
            # Update window
            window = np.append(window[1:], prediction)
            
            # Increase volatility for longer forecasts
            volatility *= 1.05
        
        return np.array(predictions), np.array(confidence), accuracy
    
    def _predict_polynomial_regression(self, features, forecast_days):
        """
        Make predictions using Polynomial Regression
        
        Args:
            features (dict): Dictionary with features
            forecast_days (int): Number of days to forecast
            
        Returns:
            tuple: Predictions, confidence intervals, accuracy metrics
        """
        # Prepare sequence data with a smaller window size
        X_train, X_test, y_train, y_test = self._prepare_train_test_data(features, window_size=5)
        
        # Reshape for Polynomial Regression
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Apply polynomial features
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train_flat)
        X_test_poly = poly.transform(X_test_flat)
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_poly, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_poly)
        accuracy = self._calculate_accuracy(y_test, y_pred)
        
        # Make future predictions
        last_window = features['close'][-5:]
        predictions = []
        confidence = []
        
        # Calculate standard error for confidence intervals
        std_error = np.std(y_test - y_pred)
        
        for _ in range(forecast_days):
            input_data = last_window.reshape(1, -1)
            input_poly = poly.transform(input_data)
            prediction = model.predict(input_poly)[0]
            
            # Calculate confidence intervals
            lower_bound = prediction - 1.96 * std_error
            upper_bound = prediction + 1.96 * std_error
            
            predictions.append(prediction)
            confidence.append([lower_bound, upper_bound])
            
            # Update window
            last_window = np.append(last_window[1:], prediction)
            
            # Increase uncertainty for future predictions
            std_error *= 1.05
        
        return np.array(predictions), np.array(confidence), accuracy
    
    def _calculate_accuracy(self, y_true, y_pred):
        """
        Calculate accuracy metrics
        
        Args:
            y_true (np.array): True values
            y_pred (np.array): Predicted values
            
        Returns:
            dict: Dictionary with accuracy metrics
        """
        # Calculate regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate directional accuracy
        direction_true = np.sign(np.diff(np.append([y_true[0]], y_true)))
        direction_pred = np.sign(np.diff(np.append([y_true[0]], y_pred)))  # Start from the same point
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'accuracy': directional_accuracy
        } 
