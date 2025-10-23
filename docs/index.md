# LSTM-BERT Stock Predictor Documentation

Welcome to the LSTM-BERT Stock Predictor documentation! This is an enterprise-ready solution for predicting stock price movements using advanced AI that combines insights from financial news with market data.

## Overview

The Stock Price Predictor leverages state-of-the-art machine learning models:
- **BERT** for text analysis of financial news and sentiment
- **LSTM** for historical trend forecasting
- **Weaviate** vector database for semantic search and data retrieval

The platform is optimized for Apple Silicon (MPS) and NVIDIA CUDA, delivering fast, efficient performance even in edge environments.

## Key Features

- Combined BERT-LSTM model for advanced stock price prediction
- Apple Silicon (MPS) GPU acceleration support
- Secure dashboard with authentication
- Real-time stock price visualization
- Vector database (Weaviate) for efficient prediction storage
- Threading support for improved performance
- HDF5 file storage for model persistence
- Interactive charts and correlation analysis
- RESTful API endpoints

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Contents:

quickstart
installation
architecture
api
models
dashboard
configuration
deployment
contributing
```

## Quick Links

- [Quick Start Guide](quickstart.md)
- [Installation Instructions](installation.md)
- [System Architecture](architecture.md)
- [API Reference](api.rst)
- [Model Documentation](models.rst)

## Requirements

- Python 3.9+
- Docker and Docker Compose
- Apple Silicon Mac (for MPS support) or any machine with CUDA support
- 8GB+ RAM recommended

## License

Copyright Protected - Permission required from the researchers.

## Support

For questions or contributions, contact: spand14@unh.newhaven.edu

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`