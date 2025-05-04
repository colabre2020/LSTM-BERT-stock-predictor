import unittest
import torch
import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.bert_lstm_model import BERTLSTMStockPredictor
from src.utils.device import get_device, get_device_properties
from src.models.trainer import ModelTrainer
from src.data.data_processor import StockDataProcessor

class TestBERTLSTMModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load configuration
        with open('config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Force CPU for testing
        cls.device = torch.device('cpu')
        print(f"\nUsing device: {cls.device}")
        
        # Initialize model with CPU device
        cls.model = BERTLSTMStockPredictor(
            num_classes=50,
            lstm_hidden_dim=64  # Smaller for testing
        )
        cls.model.device = cls.device
        cls.model.to(cls.device)
        
        # Move BERT model to CPU
        cls.model.bert.to(cls.device)
    
    def test_device_availability(self):
        """Test if MPS/CUDA is available"""
        if torch.backends.mps.is_available():
            self.assertTrue(torch.backends.mps.is_built())
        if torch.cuda.is_available():
            self.assertTrue(True)
    
    def test_model_architecture(self):
        """Test model architecture and shapes"""
        batch_size = 2
        seq_length = 32  # Smaller sequence length for testing
        num_features = 50
        
        # Create small dummy input data
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(self.device)
        attention_mask = torch.ones(batch_size, seq_length).to(self.device)
        numerical_features = torch.randn(batch_size, num_features).to(self.device)
        
        # Forward pass
        self.model.eval()  # Set to eval mode
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, numerical_features)
        
        # Check output shape
        self.assertEqual(outputs.shape, (batch_size, num_features))
        # Compare device types instead of full device objects
        self.assertEqual(outputs.device.type, self.device.type)
    
    def test_data_loading(self):
        """Test data loading and preprocessing"""
        processor = StockDataProcessor(
            self.config['data']['excel_path'],
            window_size=5,  # Small window for testing
            batch_size=2
        )
        
        data_dict, processed_data = processor.load_and_preprocess_data()
        
        self.assertIsNotNone(data_dict)
        self.assertIsNotNone(processed_data)
        self.assertTrue(isinstance(processed_data, np.ndarray))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        save_path = 'models/saved/test_model.h5'
        metadata = {'test': 'metadata'}
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_model_to_hdf5(save_path, metadata)
        
        # Create new model with same architecture
        loaded_model = BERTLSTMStockPredictor(
            num_classes=50,
            lstm_hidden_dim=64  # Match the test model
        )
        loaded_model.device = self.device
        loaded_model.to(self.device)
        
        loaded_metadata = loaded_model.load_model_from_hdf5(save_path)
        
        # Check metadata
        self.assertEqual(loaded_metadata['test'], 'metadata')
        
        # Clean up
        os.remove(save_path)
    
    def test_training_loop(self):
        """Test training loop with minimal data"""
        trainer = ModelTrainer()
        trainer.device = self.device  # Force CPU for testing
        
        # Initialize model for training test
        model = BERTLSTMStockPredictor(
            num_classes=50,
            lstm_hidden_dim=64  # Match the test model
        )
        model.device = self.device
        model.to(self.device)
        
        # Create minimal test data
        train_loader, val_loader, _ = trainer.prepare_data()
        
        # Train for just one batch
        trained_model = trainer.train(model=model, num_epochs=1, test_mode=True)
        self.assertIsNotNone(trained_model)

@unittest.skipIf(not torch.backends.mps.is_available(), "MPS not available")
class TestBERTLSTMModelMPS(TestBERTLSTMModel):
    @classmethod
    def setUpClass(cls):
        # Load configuration
        with open('config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Use MPS device
        cls.device = torch.device('mps')
        print(f"\nUsing device: {cls.device}")
        
        # Initialize model with smaller configuration for MPS
        cls.model = BERTLSTMStockPredictor(
            num_classes=50,
            lstm_hidden_dim=64  # Smaller for MPS testing
        )
        cls.model.device = cls.device
        cls.model.to(cls.device)

if __name__ == '__main__':
    unittest.main()