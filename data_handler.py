import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import streamlit as st

class CryptoDataHandler:
    """
    Handles cryptocurrency data retrieval and processing from CoinGecko API.
    Implements caching and rate limiting to comply with API restrictions.
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.request_count = 0
        self.last_request_time = time.time()
        self.rate_limit = 10  # Maximum requests per minute (free tier)
        self.rate_limit_reset = 60  # Time in seconds to reset rate limit
        
        # Top 20 cryptocurrencies by market cap (this would be dynamic in a real app)
        self.top_cryptos = {
            'bitcoin': 'Bitcoin',
            'ethereum': 'Ethereum',
            'tether': 'Tether',
            'binancecoin': 'Binance Coin',
            'solana': 'Solana',
            'xrp': 'XRP',
            'usdc': 'USD Coin',
            'cardano': 'Cardano',
            'dogecoin': 'Dogecoin',
            'polkadot': 'Polkadot',
            'tron': 'TRON',
            'chainlink': 'Chainlink',
            'polygon': 'Polygon',
            'litecoin': 'Litecoin',
            'bitcoin-cash': 'Bitcoin Cash',
            'stellar': 'Stellar',
            'uniswap': 'Uniswap',
            'monero': 'Monero',
            'cosmos': 'Cosmos',
            'ethereum-classic': 'Ethereum Classic'
        }
    
    @st.cache_data(ttl=300)
    def get_available_cryptocurrencies(self):
        """
        Returns a list of available cryptocurrencies for selection.
        Uses caching to avoid excessive API calls.
        
        Returns:
            list: List of available cryptocurrency IDs
        """
        try:
            # In a production app, this would fetch from CoinGecko API
            # However, for rate limit purposes we'll use a predefined list
            return list(self.top_cryptos.keys())
        except Exception as e:
            st.error(f"Error fetching available cryptocurrencies: {e}")
            return list(self.top_cryptos.keys())[:5]  # Return top 5 as fallback
    
    def get_crypto_name(self, crypto_id):
        """
        Get the display name for a cryptocurrency ID
        
        Args:
            crypto_id (str): Cryptocurrency ID
            
        Returns:
            str: Display name for the cryptocurrency
        """
        return self.top_cryptos.get(crypto_id, crypto_id.capitalize())
    
    def _make_api_request(self, endpoint, params=None):
        """
        Make a rate-limited API request to CoinGecko
        
        Args:
            endpoint (str): API endpoint to call
            params (dict, optional): Query parameters
            
        Returns:
            dict: API response
        """
        # Check if we need to wait due to rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if self.request_count >= self.rate_limit and time_since_last_request < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - time_since_last_request
            time.sleep(wait_time)
            self.request_count = 0
        
        # Make the API request
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            
            # Update request tracking
            self.last_request_time = time.time()
            self.request_count += 1
            
            # Check for successful response
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"API request failed with status code {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error making API request: {e}")
            return None
    
    @st.cache_data(ttl=3600)
    def get_historical_data(self, crypto_id, days):
        """
        Get historical price data for a cryptocurrency
        
        Args:
            crypto_id (str): Cryptocurrency ID
            days (int): Number of days of historical data to fetch
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        try:
            # Fetch market data from CoinGecko API
            endpoint = f"coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            data = self._make_api_request(endpoint, params)
            
            if not data:
                # Fallback to mock data if API fails
                return self._generate_mock_data(crypto_id, days)
            
            # Process the data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['volume'] = [v[1] for v in volumes]
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Generate OHLC data
            # Note: CoinGecko free API doesn't provide OHLC directly, so we estimate it
            df['close'] = df['price']
            df['open'] = df['price'].shift(1)
            df['high'] = df['price'] * (1 + np.random.uniform(0, 0.02, size=len(df)))
            df['low'] = df['price'] * (1 - np.random.uniform(0, 0.02, size=len(df)))
            
            # Fill NaN values
            df = df.fillna(method='bfill')
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return self._generate_mock_data(crypto_id, days)
    
    def _generate_mock_data(self, crypto_id, days):
        """
        Generate mock data in case the API fails
        
        Args:
            crypto_id (str): Cryptocurrency ID
            days (int): Number of days of historical data
            
        Returns:
            pd.DataFrame: DataFrame with mock historical price data
        """
        # Set seed based on crypto_id for consistent mock data
        seed = sum(ord(c) for c in crypto_id)
        np.random.seed(seed)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate mock price data with realistic volatility
        if crypto_id == 'bitcoin':
            base_price = 30000
            volatility = 0.03
        elif crypto_id == 'ethereum':
            base_price = 2000
            volatility = 0.04
        else:
            base_price = np.random.uniform(0.1, 500)
            volatility = np.random.uniform(0.02, 0.08)
        
        # Generate price with random walk
        returns = np.random.normal(0, volatility, size=len(date_range))
        price_factor = (1 + returns).cumprod()
        close_prices = base_price * price_factor
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        df['close'] = close_prices
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, volatility/2, size=len(df)))
        df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, volatility/2, size=len(df))))
        df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, volatility/2, size=len(df))))
        df['volume'] = np.random.uniform(base_price * 1000, base_price * 10000, size=len(df))
        
        # Fill first row NaN values
        df = df.fillna(method='bfill')
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        return df
    
    def _add_technical_indicators(self, df):
        """
        Add technical indicators to the DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Volatility (using standard deviation of returns)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        
        return df
    
    @st.cache_data(ttl=300)
    def get_market_cap(self, crypto_id):
        """
        Get market cap for a cryptocurrency
        
        Args:
            crypto_id (str): Cryptocurrency ID
            
        Returns:
            float: Market cap value
        """
        try:
            endpoint = f"coins/{crypto_id}"
            data = self._make_api_request(endpoint)
            
            if data and 'market_data' in data:
                return data['market_data']['market_cap']['usd']
            
            # Fallback to mock data
            if crypto_id == 'bitcoin':
                return 600000000000  # $600B
            elif crypto_id == 'ethereum':
                return 240000000000  # $240B
            else:
                # Generate a random market cap based on crypto_id
                seed = sum(ord(c) for c in crypto_id)
                np.random.seed(seed)
                return np.random.randint(100000000, 10000000000)
                
        except Exception as e:
            st.error(f"Error fetching market cap: {e}")
            # Return a reasonable mock value
            return 1000000000  # $1B
    
    @st.cache_data(ttl=300)
    def get_24h_volume(self, crypto_id):
        """
        Get 24h trading volume for a cryptocurrency
        
        Args:
            crypto_id (str): Cryptocurrency ID
            
        Returns:
            float: 24h volume
        """
        try:
            endpoint = f"coins/{crypto_id}"
            data = self._make_api_request(endpoint)
            
            if data and 'market_data' in data:
                return data['market_data']['total_volume']['usd']
            
            # Fallback to mock data
            if crypto_id == 'bitcoin':
                return 30000000000  # $30B
            elif crypto_id == 'ethereum':
                return 15000000000  # $15B
            else:
                # Generate a random volume based on crypto_id
                seed = sum(ord(c) for c in crypto_id)
                np.random.seed(seed)
                return np.random.randint(10000000, 2000000000)
                
        except Exception as e:
            st.error(f"Error fetching 24h volume: {e}")
            # Return a reasonable mock value
            return 50000000  # $50M
