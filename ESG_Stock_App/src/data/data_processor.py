import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from concurrent.futures import ThreadPoolExecutor
import h5py
from sklearn.preprocessing import MinMaxScaler
import threading
from typing import Dict, List, Tuple
import os

class StockDataProcessor:
    def __init__(self, excel_path: str, window_size: int = 42, batch_size: int = 32):
        self.excel_path = excel_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.scaler = MinMaxScaler()
        self.lock = threading.Lock()
        
    def load_and_preprocess_data(self) -> Tuple[Dict, np.ndarray]:
        # Load Excel data
        df = pd.read_excel(self.excel_path)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        
        # Get stock names and prepare numerical data
        stock_names = df.columns[1:].tolist()
        numerical_data = df[stock_names].values
        
        # Normalize numerical data in parallel
        with ThreadPoolExecutor() as executor:
            normalized_chunks = list(executor.map(
                self._normalize_chunk, 
                np.array_split(numerical_data, os.cpu_count())
            ))
        normalized_data = np.vstack(normalized_chunks)
        
        return {'dates': df['Date'].values, 'stock_names': stock_names}, normalized_data
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        # Process each chunk of data
        return chunk.copy()
    
    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        with self.lock:
            return self.scaler.fit_transform(chunk)
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training, ensuring proper batch handling"""
        # Calculate optimal chunk size based on available memory
        chunk_size = min(1000, len(data) - self.window_size)
        chunks = range(0, len(data) - self.window_size, chunk_size)
        
        sequences = []
        
        def process_chunk(start_idx):
            chunk_sequences = []
            chunk_end = min(start_idx + chunk_size, len(data) - self.window_size)
            
            for i in range(start_idx, chunk_end):
                # Use first column as target (closing price)
                sequence = data[i:i + self.window_size]
                target = data[i + self.window_size, 0]  # Take only first feature as target
                chunk_sequences.append({
                    'X': sequence,
                    'y': target
                })
            return chunk_sequences
        
        # Process sequences in parallel
        with ThreadPoolExecutor() as executor:
            all_sequences = list(executor.map(process_chunk, chunks))
        
        # Flatten sequences
        sequences = [s for chunk in all_sequences for s in chunk]
        
        # Convert to numpy arrays
        X = np.array([s['X'] for s in sequences])
        y = np.array([s['y'] for s in sequences])
        
        return X, y
    
    def save_to_hdf5(self, filepath: str, data_dict: Dict, processed_data: np.ndarray):
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.create_dataset('dates', data=data_dict['dates'].astype('S'))
            meta_group.create_dataset('stock_names', data=np.array(data_dict['stock_names'], dtype='S'))
            
            # Save processed data
            f.create_dataset('processed_data', data=processed_data)
            f.create_dataset('scaler_params', data=np.array([
                self.scaler.data_min_,
                self.scaler.data_max_,
                self.scaler.scale_,
                self.scaler.min_
            ]))
    
    def load_from_hdf5(self, filepath: str) -> Tuple[Dict, np.ndarray]:
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            data_dict = {
                'dates': f['metadata/dates'][:].astype('datetime64'),
                'stock_names': f['metadata/stock_names'][:].astype(str)
            }
            
            # Load processed data
            processed_data = f['processed_data'][:]
            
            # Restore scaler
            scaler_params = f['scaler_params'][:]
            self.scaler.data_min_ = scaler_params[0]
            self.scaler.data_max_ = scaler_params[1]
            self.scaler.scale_ = scaler_params[2]
            self.scaler.min_ = scaler_params[3]
            
            return data_dict, processed_data

class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, tokenizer: BertTokenizer):
        self.X = torch.FloatTensor(X)
        # Ensure y is a column vector [batch_size, 1]
        self.y = torch.FloatTensor(y).unsqueeze(-1)
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict:
        # Convert numerical data to text for BERT
        text_input = f"Stock price sequence: {self.X[idx].tolist()}"
        
        # Tokenize text
        encoded = self.tokenizer(
            text_input,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'numerical_features': self.X[idx],
            'labels': self.y[idx]  # Shape: [1]
        }