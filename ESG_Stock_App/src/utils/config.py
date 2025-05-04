# Configuration settings for the ESG stock predictor application

# Model parameters
MODEL_NAME = "bert"  # Options: "bert", "lstm"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# File paths
STOCK_DATA_PATH = "data/STOCK_PRICE_TOP_50.xlsx"
TRAINING_DATA_PATH = "data/training_data.csv"
TESTING_DATA_PATH = "data/testing_data.csv"

# GPU settings
USE_GPU = True  # Set to True to use Apple MPS GPU for training

# Logging settings
LOGGING_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"