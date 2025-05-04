def preprocess_data(data):
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    
    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

def split_data(data, train_size=0.8):
    # Split the data into training and testing sets
    train_size = int(len(data) * train_size)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data

def transform_features(data):
    # Example transformation function
    # This can be customized based on feature engineering needs
    return data  # Placeholder for actual transformation logic