import torch
import torch.nn as nn
from transformers import BertModel
import threading
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
import io
import logging
from ..utils.device import clear_memory
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

logger = logging.getLogger(__name__)

class BERTLSTMStockPredictor(nn.Module):
    def __init__(self, num_classes=None, lstm_hidden_dim=128):
        super(BERTLSTMStockPredictor, self).__init__()
        
        # Initialize device first
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                 else 'cuda' if torch.cuda.is_available() 
                                 else 'cpu')
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.gradient_checkpointing_enable()
        self._freeze_bert_layers()

        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.num_classes = num_classes or 1  # Default to 1 if not provided

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_dim * 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        total_hidden_dim = self.lstm_hidden_dim * 2

        self.fc1 = nn.Sequential(
            nn.Linear(total_hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Initialize fc2 with 1 output by default
        self.fc2 = nn.Linear(256, 1)

        self.scaler = GradScaler()
        self.num_projection = None
        
        # Move model to device
        self.to(self.device)
    
    def _freeze_bert_layers(self):
        """Freeze first few BERT layers to reduce memory and computation"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(8):  # Freeze first 8 layers
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, numerical_features=None):
        batch_size = input_ids.size(0)
        
        # Skip autocast for MPS, use it only for CUDA
        context = autocast() if self.device.type == 'cuda' else nullcontext()
        with context:
            if self.device.type == 'mps':
                clear_memory(self.device)
            
            # Process BERT in chunks for memory efficiency
            chunk_size = min(16, batch_size)  # Ensure chunk_size doesn't exceed batch_size
            bert_outputs = []
            
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_input_ids = input_ids[i:end_idx]
                chunk_attention_mask = attention_mask[i:end_idx]
                
                with torch.no_grad():  # Use cached BERT outputs
                    chunk_output = self.bert(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                bert_outputs.append(chunk_output.last_hidden_state)
            
            sequence_output = torch.cat(bert_outputs, dim=0)
            
            # Process LSTM with attention
            lstm_out, _ = self.lstm(sequence_output)
            
            # Apply attention mechanism
            attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Global average pooling
            pooled = torch.mean(attn_output, dim=1)
            
            # Handle numerical features
            if numerical_features is not None:
                # Flatten numerical features if they are 3D
                if numerical_features.dim() == 3:
                    numerical_features = numerical_features.view(batch_size, -1)
                
                if self.num_projection is None or self.num_projection.out_features != pooled.shape[-1]:
                    # Adjust projection to match pooled output size
                    self.num_projection = nn.Linear(
                        numerical_features.shape[-1], 
                        pooled.shape[-1]
                    ).to(self.device)
                numerical_features = self.num_projection(numerical_features)
                pooled = pooled + numerical_features
            
            # Final prediction layers
            x = self.fc1(pooled)
            output = self.fc2(x)  # Shape: [batch_size, num_classes]
            
            return output

    def train_step(self, batch, optimizer, criterion):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device).view(-1, 1)  # Ensure labels are [batch_size, 1]
        numerical_features = batch.get('numerical_features')
        
        if numerical_features is not None:
            numerical_features = numerical_features.to(self.device)
        
        # Skip autocast for MPS, use it only for CUDA
        context = autocast() if self.device.type == 'cuda' else nullcontext()
        with context:
            outputs = self(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs, labels)
        
        # Scale loss and backward pass only for CUDA
        if self.device.type == 'cuda':
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if self.device.type == 'mps':
            clear_memory(self.device)
        
        return loss.item()

    def save_model_to_hdf5(self, filepath, metadata=None):
        """Save model state and metadata to HDF5 file"""
        # Clear memory before saving if using MPS
        if self.device.type == 'mps':
            clear_memory(self.device)
            
        with h5py.File(filepath, 'w') as f:
            # Save model architecture parameters
            f.attrs['lstm_hidden_dim'] = self.lstm_hidden_dim
            if self.num_projection is not None:
                f.attrs['num_features_dim'] = self.num_projection.in_features
            
            # Save model state
            state_dict_bytes = self._serialize_state_dict()
            f.create_dataset('model_state', data=np.void(state_dict_bytes))
            
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    meta_group.attrs[key] = value

    def load_model_from_hdf5(self, filepath):
        """Load model state and metadata from HDF5 file"""
        # Clear memory before loading if using MPS
        if self.device.type == 'mps':
            clear_memory(self.device)
            
        with h5py.File(filepath, 'r') as f:
            # Load architecture parameters and ensure correct types
            self.lstm_hidden_dim = int(f.attrs['lstm_hidden_dim'])
            if 'num_features_dim' in f.attrs:
                num_features_dim = int(f.attrs['num_features_dim'])
                self.num_projection = nn.Linear(
                    num_features_dim, 
                    self.lstm_hidden_dim * 2
                ).to(self.device)
            
            # Load model state with proper device mapping
            state_dict_bytes = f['model_state'][()].tobytes()
            state_dict = torch.load(
                io.BytesIO(state_dict_bytes), 
                map_location=self.device
            )
            
            # Adjust fc2 layer dynamically based on saved model
            if 'fc2.weight' in state_dict:
                saved_output_size = state_dict['fc2.weight'].size(0)
                if saved_output_size != self.fc2.out_features:
                    self.fc2 = nn.Linear(256, saved_output_size).to(self.device)

            self.load_state_dict(state_dict, strict=False)
            
            if 'metadata' in f:
                return dict(f['metadata'].attrs)
            return None

    def _serialize_state_dict(self):
        """Serialize model state dict to bytes"""
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.getvalue()

    def train_threaded(self, train_dataloader, optimizer, criterion, num_epochs, num_threads=4):
        def train_batch(batch):
            return self.train_step(batch, optimizer, criterion)

        self.train()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for epoch in range(num_epochs):
                losses = list(executor.map(train_batch, train_dataloader))
                epoch_loss = sum(losses) / len(losses)
                optimizer.step()
                
                # Clear memory after each epoch if using MPS
                if self.device.type == 'mps':
                    clear_memory(self.device)
                
                if epoch % 10 == 0:
                    logger.info(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')

    def predict(self, input_ids, attention_mask, numerical_features=None):
        self.eval()
        with torch.no_grad():
            # Clear memory before prediction if using MPS
            if self.device.type == 'mps':
                clear_memory(self.device)
                
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if numerical_features is not None:
                numerical_features = numerical_features.to(self.device)
            
            outputs = self(input_ids, attention_mask, numerical_features)
            return outputs