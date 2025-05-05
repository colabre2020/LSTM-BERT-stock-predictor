# Stock Price Prediction RAG Application with BERT + LSTM Models

Stock Price Predictor is a next-generation, enterprise-ready solution for predicting stock price movements using advanced AI. By combining insights from financial news with market data, it leverages state-of-the-art machine learning — specifically BERT for text analysis and LSTM for historical trend forecasting — to generate high-confidence predictions. Optimized for Apple’s latest hardware, the platform delivers fast, efficient performance, even in edge environments. A secure, user-friendly dashboard offers real-time analytics and intuitive visualizations. Integrated with a vector database, the system also supports intelligent information retrieval, making it valuable for investment research, risk assessment, and strategic decision-making.

# Solution Architecture

![image](https://github.com/user-attachments/assets/407f8c4b-9295-4bfe-951a-dc8592e35a65)



# Application Mocks


![image](https://github.com/user-attachments/assets/95b39d3e-6a47-40e1-a85e-9aa58b48a7cd)


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
LSTM-BERT stock-predictor/
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
Copyright Protected, need permission from the researchers

## Contributing

If you want to contribute to this project, let's know.

## Reference

BERT - https://doi.org/10.48550/arXiv.1810.04805
LSTM - https://doi.org/10.1162/neco.1997.9.8.1735
https://doi.org/10.1080/20430795.2024.2377551 
Weaviate - https://weaviate.io/blog 
LSTM - https://developer.nvidia.com/discover/lstm#:~:text=A%20Long%20short%2Dterm%20memory,cycles%20through%20the%20feedback%20loops 

