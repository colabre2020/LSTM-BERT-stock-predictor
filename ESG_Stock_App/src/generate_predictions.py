import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.db_handler import WeaviateHandler
import os
import logging
import sys
import traceback

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generate_predictions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate sample stock data if the Excel file can't be read"""
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    data = {'Date': dates}
    for stock in stocks:
        # Generate realistic looking stock prices
        base_price = np.random.uniform(100, 500)
        volatility = np.random.uniform(0.01, 0.03)
        prices = [base_price]
        for _ in range(29):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        data[stock] = prices
    
    return pd.DataFrame(data)

def generate_initial_predictions():
    try:
        # Try to load real data first
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            excel_path = os.path.join(os.path.dirname(current_dir), 'data', 'STOCK_PRICE_TOP_50.xlsx')
            logger.info(f"Attempting to read stock data from: {excel_path}")
            df = pd.read_excel(excel_path)
        except Exception as e:
            logger.warning(f"Could not read Excel file: {str(e)}")
            logger.info("Generating sample data instead")
            df = generate_sample_data()
        
        logger.info(f"Successfully loaded/generated data. Shape: {df.shape}")
        logger.info(f"Columns in DataFrame: {df.columns.tolist()}")
        
        # Get stock names from columns (excluding the Date column)
        stocks = [col for col in df.columns if col != 'Date']
        logger.info(f"Found {len(stocks)} stocks: {stocks}")
        
        # Initialize components
        logger.info("Initializing Weaviate connection...")
        db_handler = WeaviateHandler()
        
        # Generate predictions for each stock
        for stock in stocks:
            try:
                logger.info(f"Processing stock: {stock}")
                # Get actual stock prices
                stock_prices = df[stock].values[-30:]  # Last 30 days of actual prices
                
                # Generate predictions for next 7 days
                for i in range(7):
                    date = datetime.now() + timedelta(days=i)
                    
                    # Generate a realistic prediction based on last price
                    last_price = float(stock_prices[-1])
                    # Use smaller random variation for more realistic predictions
                    prediction = float(np.random.uniform(0.98, 1.02) * last_price)
                    confidence = float(np.random.uniform(0.85, 0.95))
                    
                    logger.info(f"Storing prediction for {stock} on {date.date()}: {prediction:.2f}")
                    # Store prediction
                    try:
                        db_handler.store_prediction(
                            stock_name=stock,
                            prediction=prediction,
                            confidence=confidence,
                            actual_price=last_price,
                            metadata={"generation_date": date.isoformat()}
                        )
                        logger.info(f"Successfully stored prediction for {stock} on {date.date()}")
                    except Exception as e:
                        logger.error(f"Failed to store prediction in Weaviate: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
                    
            except Exception as e:
                logger.error(f"Error generating predictions for stock {stock}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                
    except Exception as e:
        logger.error(f"Error in generate_initial_predictions: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    logger.info("Starting prediction generation script...")
    generate_initial_predictions()
    logger.info("Finished prediction generation script.")