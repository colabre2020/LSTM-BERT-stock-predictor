import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, Tuple, Optional
import yaml
import os
from tqdm import tqdm
import logging
from datetime import datetime

from ..data.data_processor import StockDataProcessor, StockDataset
from .bert_lstm_model import BERTLSTMStockPredictor
from ..utils.device import get_device, clear_memory, memory_status

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = get_device(config_path)
        self.lock = threading.Lock()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Using device: {self.device}")
        if self.device.type in ['cuda', 'mps']:
            logger.info(f"Device properties: {memory_status(self.device)}")
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        processor = StockDataProcessor(
            self.config['data']['excel_path'],
            self.config['data']['window_size'],
            self.config['data']['batch_size']
        )
        
        # Use more threads for data loading
        thread_count = min(os.cpu_count() * 2, 16)
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            data_future = executor.submit(processor.load_and_preprocess_data)
            data_dict, processed_data = data_future.result()
        
        X, y = processor.create_sequences(processed_data)
        
        # Split data with optimized ratios
        total_samples = len(X)
        train_size = int(total_samples * self.config['data']['train_split'])
        val_size = int(total_samples * self.config['data']['val_split'])
        
        # Create datasets with efficient memory handling
        datasets = {
            'train': StockDataset(X[:train_size], y[:train_size], processor.tokenizer),
            'val': StockDataset(X[train_size:train_size + val_size], y[train_size:train_size + val_size], processor.tokenizer),
            'test': StockDataset(X[train_size + val_size:], y[train_size + val_size:], processor.tokenizer)
        }
        
        # Optimize number of workers and pin memory
        num_workers = min(os.cpu_count(), 8)
        pin_memory = self.device.type != 'cpu'
        batch_size = self.config['data']['batch_size']
        
        dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=2 if pin_memory else None
            )
            for split, dataset in datasets.items()
        }
        
        return dataloaders['train'], dataloaders['val'], dataloaders['test']
    
    def train(self, model: Optional[BERTLSTMStockPredictor] = None, num_epochs: Optional[int] = None, test_mode: bool = False) -> BERTLSTMStockPredictor:
        train_loader, val_loader, _ = self.prepare_data()
        
        if model is None:
            model = BERTLSTMStockPredictor(
                num_classes=1,  # Changed to 1 for single value prediction
                lstm_hidden_dim=self.config['model']['lstm']['hidden_size']
            )
        
        model = model.to(self.device)
        
        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['model']['training']['learning_rate'],
            weight_decay=0.01
        )
        
        # Use cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
        
        criterion = nn.MSELoss()
        
        if test_mode:
            num_epochs = 1
            train_loader = self._create_test_loader(train_loader.dataset)
            val_loader = self._create_test_loader(val_loader.dataset)
        else:
            num_epochs = num_epochs or self.config['model']['training']['epochs']
        
        best_val_loss = float('inf')
        patience = self.config['model']['training']['early_stopping_patience']
        patience_counter = 0
        
        model.train()
        for epoch in range(num_epochs):
            train_losses = []
            
            # Training phase with progress bar
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    loss = self._process_batch(model, batch, optimizer, criterion, is_training=True)
                    train_losses.append(loss)
                    
                    # Update progress bar
                    avg_loss = np.mean(train_losses[-100:])
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
                    
                    if test_mode and batch_idx >= 1:
                        break
            
            # Validation phase with batched processing
            if not test_mode:
                val_losses = []
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        val_loss = self._process_batch(model, batch, optimizer, criterion, is_training=False)
                        val_losses.append(val_loss)
                model.train()
                
                val_loss = np.mean(val_losses)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model(model, {
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_loss': np.mean(train_losses)
                    })
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                        break
                
                logger.info(f'Epoch {epoch + 1} - Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}')
        
        return model
    
    def _process_batch(self, model: BERTLSTMStockPredictor, batch: Dict, optimizer: torch.optim.Optimizer, criterion: nn.Module, is_training: bool = True) -> float:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device).view(-1, 1)
        numerical_features = batch.get('numerical_features')
        
        if numerical_features is not None:
            numerical_features = numerical_features.to(self.device)
        
        if is_training:
            optimizer.zero_grad()
            
        outputs = model(input_ids, attention_mask, numerical_features)
        loss = criterion(outputs, labels)
        
        if is_training:
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def _validate_batched(self, model: BERTLSTMStockPredictor, val_loader: DataLoader, criterion: nn.Module) -> float:
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).view(-1, 1)
                numerical_features = batch.get('numerical_features')
                
                if numerical_features is not None:
                    numerical_features = numerical_features.to(self.device)
                
                outputs = model(input_ids, attention_mask, numerical_features)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
        
        model.train()
        return np.mean(val_losses)
    
    def _create_test_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
    
    def save_model(self, model: BERTLSTMStockPredictor, metadata: Dict):
        os.makedirs(os.path.dirname(self.config['model']['save_path']), exist_ok=True)
        model.save_model_to_hdf5(self.config['model']['save_path'], metadata)
        logger.info(f"Model saved to {self.config['model']['save_path']}")
    
    def load_model(self) -> Tuple[BERTLSTMStockPredictor, Dict]:
        """Load the model with correct initialization parameters"""
        model = BERTLSTMStockPredictor(
            num_classes=1,  # Initialize with 1 for single value prediction
            lstm_hidden_dim=self.config['model']['lstm']['hidden_size']
        ).to(self.device)
        
        metadata = model.load_model_from_hdf5(self.config['model']['save_path'])
        logger.info(f"Model loaded from {self.config['model']['save_path']}")
        return model, metadata

def train_model():
    trainer = ModelTrainer()
    model = trainer.train()
    return model

if __name__ == '__main__':
    train_model()