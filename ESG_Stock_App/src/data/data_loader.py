def load_stock_data(file_path):
    import pandas as pd

    # Load the stock data from the Excel file
    data = pd.read_excel(file_path)

    # Convert the data into a suitable format for training
    # Assuming the data has columns 'Date' and 'Price'
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    return data

def preprocess_data(data):
    # Handle missing values
    data.fillna(method='ffill', inplace=True)

    # Normalize the stock prices
    data['Price'] = (data['Price'] - data['Price'].mean()) / data['Price'].std()

    return data

def split_data(data, train_size=0.8):
    # Split the data into training and testing sets
    train_size = int(len(data) * train_size)
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data

# Example usage
if __name__ == "__main__":
    stock_data = load_stock_data('STOCK_PRICE_TOP_50.xlsx')
    processed_data = preprocess_data(stock_data)
    train_data, test_data = split_data(processed_data)