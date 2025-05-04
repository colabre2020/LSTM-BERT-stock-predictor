import torch
import yaml
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_device(config_path: str = 'config.yaml') -> torch.device:
    """
    Get the appropriate device for PyTorch based on availability and config.
    Handles fallback logic for MPS -> CUDA -> CPU.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        device_config = config.get('device', {})
        preferred_device = device_config.get('type', 'mps')
        fallback_order = device_config.get('fallback_order', ['mps', 'cuda', 'cpu'])
        
        # Try devices in order of preference
        for device_type in fallback_order:
            if device_type == 'mps' and torch.backends.mps.is_available():
                logger.info("Using MPS (Apple Silicon) device")
                return torch.device('mps')
            elif device_type == 'cuda' and torch.cuda.is_available():
                logger.info("Using CUDA device")
                return torch.device('cuda')
            elif device_type == 'cpu':
                logger.info("Using CPU device")
                return torch.device('cpu')
        
        # If no device in fallback order is available, default to CPU
        logger.warning("No preferred devices available, falling back to CPU")
        return torch.device('cpu')
        
    except Exception as e:
        logger.error(f"Error determining device: {str(e)}")
        logger.warning("Defaulting to CPU due to error")
        return torch.device('cpu')

def get_device_properties() -> dict:
    """
    Get properties of the current device for logging and monitoring.
    """
    device = get_device()
    properties = {
        'device_type': device.type,
        'device_index': device.index if hasattr(device, 'index') else None,
    }
    
    if device.type == 'cuda':
        properties.update({
            'name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated(0),
            'memory_reserved': torch.cuda.memory_reserved(0),
            'max_memory_allocated': torch.cuda.max_memory_allocated(0)
        })
    elif device.type == 'mps':
        properties.update({
            'name': 'Apple Silicon GPU',
            'is_available': torch.backends.mps.is_available(),
            'is_built': torch.backends.mps.is_built()
        })
    
    return properties

def memory_status(device: Optional[torch.device] = None) -> dict:
    """
    Get current memory status of the device.
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        return {
            'allocated': torch.cuda.memory_allocated(device),
            'reserved': torch.cuda.memory_reserved(device),
            'max_allocated': torch.cuda.max_memory_allocated(device)
        }
    elif device.type == 'mps':
        # MPS doesn't provide memory stats, return placeholder
        return {
            'allocated': 'N/A',
            'reserved': 'N/A',
            'max_allocated': 'N/A'
        }
    else:
        return {
            'allocated': 'N/A',
            'reserved': 'N/A',
            'max_allocated': 'N/A'
        }

def clear_memory(device: Optional[torch.device] = None):
    """
    Clear memory on the device if possible.
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == 'mps':
        # MPS doesn't need explicit memory clearing
        pass
    
    # Force garbage collection
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()