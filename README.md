# Crypto Price Predictor

A comprehensive system for forecasting cryptocurrency prices (BTC, ETH, and altcoins) using machine learning and deep learning models. The system integrates real-time APIs, technical analysis, and ML-based predictions with a modular, extensible architecture.

## Features

- **Multi-source data collection** from Binance, CoinGecko, and Fear & Greed Index APIs
- **Automated historical and real-time data fetching** with robust error handling
- **Data validation and cleaning pipeline** for ensuring data quality
- **Database storage** with SQLAlchemy supporting both SQLite and PostgreSQL
- **Modular codebase** designed for easy extension and maintenance
- **Scalable architecture** supporting multiple cryptocurrency pairs

## Project Structure

```
src/
└── data/
    ├── collectors.py      # API integrations for data fetching
    ├── preprocessors.py   # Data cleaning and validation
    └── storage.py         # Database operations and management
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NoLongerHumanHQ/crypto-price-predictor.git
   cd crypto-price-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database configuration
   ```

## Usage

### Basic Usage

The system can be used as individual modules or integrated into a larger pipeline:

```python
from src.data.collectors import BinanceCollector, CoinGeckoCollector
from src.data.preprocessors import DataPreprocessor
from src.data.storage import CryptoDataStorage

# Initialize data collector
collector = BinanceCollector()
data = collector.fetch_historical_data('BTCUSDT', '1d', 100)

# Process the data
preprocessor = DataPreprocessor()
cleaned_data = preprocessor.clean_data(data)

# Store in database
storage = CryptoDataStorage()
storage.save_data(cleaned_data)
```

### Running Data Collection

Run the data collection modules as standalone scripts:

```bash
python src/data/collectors.py
```

Or import them into your custom pipeline for more advanced usage.

## Core Components

### Data Collectors (`collectors.py`)

Contains classes for fetching data from various cryptocurrency APIs:

- **BinanceCollector**: Fetches OHLCV data from Binance exchange
- **CoinGeckoCollector**: Retrieves market data and metadata from CoinGecko
- **FearGreedCollector**: Collects market sentiment data from Fear & Greed Index

### Data Preprocessor (`preprocessors.py`)

The `DataPreprocessor` class handles:

- Data cleaning and validation
- Outlier detection and removal
- Missing value imputation
- Data normalization and standardization
- Feature engineering preparation

### Data Storage (`storage.py`)

The `CryptoDataStorage` class provides:

- Database connection management with SQLAlchemy
- Data persistence for both SQLite and PostgreSQL
- Efficient data retrieval and querying
- Data backup and recovery operations

## Configuration

Create a `.env` file based on `.env.example` with the following variables:

```env
# API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
COINGECKO_API_KEY=your_coingecko_api_key

# Database Configuration
DATABASE_URL=sqlite:///crypto_data.db
# Or for PostgreSQL:
# DATABASE_URL=postgresql://username:password@localhost:5432/crypto_db

# Data Collection Settings
DEFAULT_SYMBOL=BTCUSDT
DEFAULT_INTERVAL=1d
DEFAULT_LIMIT=1000
```

## Requirements

- Python 3.8+
- SQLAlchemy
- Requests
- Pandas
- NumPy
- python-dotenv

See `requirements.txt` for complete dependency list.

## Roadmap

### Phase 1: Foundation (Current)
- [x] Multi-source data collection
- [x] Data preprocessing pipeline
- [x] Database storage system

### Phase 2: Analytics & Modeling
- [ ] Feature engineering (technical indicators, sentiment analysis)
- [ ] Machine learning model development (LSTM, Random Forest, XGBoost)
- [ ] Deep learning models (CNN-LSTM, Transformer networks)
- [ ] Model evaluation and backtesting framework

### Phase 3: Deployment & Interface
- [ ] REST API development
- [ ] Web dashboard with real-time predictions
- [ ] Model serving infrastructure
- [ ] Automated trading integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new functionality
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## Architecture

The system follows a modular architecture pattern:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processors    │    │    Storage      │
│                 │    │                 │    │                 │
│ • Binance API   │───▶│ • Data Cleaning │───▶│ • SQLAlchemy    │
│ • CoinGecko API │    │ • Validation    │    │ • SQLite/Postgres│
│ • Fear & Greed  │    │ • Normalization │    │ • Data Models   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Performance Considerations

- **Rate Limiting**: Built-in API rate limiting to respect exchange limits
- **Caching**: Intelligent caching to reduce API calls
- **Batch Processing**: Efficient batch operations for large datasets
- **Memory Management**: Optimized data structures for large time series

## Troubleshooting

### Common Issues

**API Connection Errors**
- Verify API keys are correctly configured
- Check internet connectivity
- Ensure API rate limits are not exceeded

**Database Connection Issues**
- Verify database URL format
- Check database permissions
- Ensure database service is running

**Data Quality Issues**
- Review data preprocessing settings
- Check for API data format changes
- Validate date ranges and symbols

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves significant risk, and past performance does not guarantee future results. Always conduct your own research and consider consulting with financial advisors before making investment decisions.

## Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation wiki
- Review existing discussions and PRs

---

**Note**: This project is actively under development. Features and APIs may change between versions. Please check the changelog for updates.

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/NoLongerHumanHQ">NoLongerHumanHQ</a></sub>
</div>
