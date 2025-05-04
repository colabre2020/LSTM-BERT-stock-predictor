# Stock Price Prediction System with BERT-LSTM

An enterprise-grade stock price prediction system that combines BERT and LSTM models, optimized for Apple Silicon (MPS) GPU acceleration, with a secure dashboard and vector database integration.

## Features

- Combined BERT-LSTM model for advanced stock price prediction
- Apple Silicon (MPS) GPU acceleration support
- Secure dashboard with authentication
- Real-time stock price visualization
- Vector database (Weaviate) for efficient prediction storage
- Threading support for improved performance
- HDF5 file storage for model persistence
- Interactive charts and correlation analysis
- RESTful API endpoints

## Requirements

- Python 3.9+
- Docker and Docker Compose
- Apple Silicon Mac (for MPS support) or any machine with CUDA support
- 8GB+ RAM recommended

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd esg-stock-predictor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the Docker containers:
```bash
cd docker
docker-compose up -d
```

5. Access the dashboard:
- Open http://localhost:5000 in your browser
- Default login credentials:
  - Username: admin
  - Password: admin

## Architecture

The system consists of several components:

- **BERT-LSTM Model**: Combines BERT's language understanding with LSTM's sequential prediction
- **Weaviate Database**: Stores predictions and enables similarity search
- **Dashboard**: Secure web interface for visualization and analysis
- **API**: RESTful endpoints for model interaction

## Configuration

Edit `config.yaml` to customize:

- Model parameters
- Training settings
- Database connections
- Dashboard settings
- Security options

## Development

### Directory Structure

```
esg-stock-predictor/
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── data/                # Data storage
├── docker/              # Docker configuration
├── models/              # Model artifacts
└── src/                # Source code
    ├── api/            # API endpoints
    ├── dashboard/      # Web interface
    ├── data/          # Data processing
    ├── models/        # ML models
    └── utils/         # Utilities
```

### Running Tests

```bash
python -m pytest tests/
```

### API Documentation

#### Authentication

```bash
# Login
POST /api/auth/login
{
    "username": "admin",
    "password": "admin"
}

# Response
{
    "token": "jwt-token"
}
```

#### Predictions

```bash
# Get predictions
GET /api/predictions/<stock_name>?days=30

# Make prediction
POST /api/predict
{
    "stock_name": "AAPL"
}
```

## Model Training

The system uses a combined BERT-LSTM architecture:

1. BERT processes textual data and market sentiment
2. LSTM handles time-series prediction
3. Both models are combined for final prediction

To train the model:

```bash
python -m src.models.trainer
```

## Dashboard Features

- Real-time stock price visualization
- Model performance metrics
- Stock correlation analysis
- Prediction accuracy tracking
- User authentication and session management

## Security

- JWT-based authentication
- Password hashing with bcrypt
- Secure session management
- API endpoint protection

## Production Deployment

For production deployment:

1. Update `config.yaml` with production settings
2. Set secure passwords and API keys
3. Enable HTTPS
4. Configure proper database backups
5. Set up monitoring and logging

## License

[Your License]

## Contributing

[Contributing Guidelines]
