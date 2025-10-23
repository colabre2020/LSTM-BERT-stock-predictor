# Quick Start Guide

Get up and running with the LSTM-BERT Stock Predictor in just a few steps.

## Prerequisites

Before you begin, ensure you have:
- Python 3.9 or higher
- Docker and Docker Compose installed
- Git installed
- At least 8GB RAM

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/colabre2020/LSTM-BERT-stock-predictor.git
cd LSTM-BERT-stock-predictor
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Docker Services

```bash
cd docker
docker-compose up -d
```

### 5. Access the Dashboard

Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

**Default Login Credentials:**
- Username: `admin`
- Password: `admin`

## First Prediction

Once logged in, you can:

1. Select a stock symbol (e.g., AAPL, GOOGL, TSLA)
2. Choose the prediction timeframe
3. Click "Generate Prediction"
4. View the results in interactive charts

## Next Steps

- [Learn about the Architecture](architecture.md)
- [Explore the API](api.rst)
- [Configure the System](configuration.md)
- [Deploy to Production](deployment.md)

## Troubleshooting

### Common Issues

**Docker containers won't start:**
- Ensure Docker is running
- Check if ports 5000 and 8080 are available
- Run `docker-compose logs` to see error messages

**Installation errors:**
- Make sure you're using Python 3.9+
- Try upgrading pip: `pip install --upgrade pip`
- On Apple Silicon Macs, ensure you have the correct architecture packages

**GPU acceleration not working:**
- For Apple Silicon: Ensure you have PyTorch with MPS support
- For NVIDIA: Verify CUDA installation and compatible PyTorch version

## Support

If you encounter issues:
1. Check the [troubleshooting section](troubleshooting.md)
2. Review the [configuration guide](configuration.md)
3. Contact support: spand14@unh.newhaven.edu