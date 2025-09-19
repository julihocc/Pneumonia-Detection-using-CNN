"""
Utility functions for the pneumonia detection project
"""
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For GPU determinism (if available)
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.enable_op_determinism()


def setup_gpu():
    """Setup GPU configuration for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # Set visible devices
            tf.config.set_visible_devices(gpus, 'GPU')
            
            print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPUs found. Using CPU.")


def create_directories(paths: list):
    """Create directories if they don't exist"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_model_size(model):
    """Get model size in MB"""
    param_size = sum([tf.keras.utils.count_params(w) for w in model.weights])
    # Assume float32 (4 bytes per parameter)
    return param_size * 4 / (1024 * 1024)


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def get_system_info() -> dict:
    """Get system information"""
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': format_bytes(psutil.virtual_memory().total),
        'tensorflow_version': tf.__version__,
    }
    
    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        info['gpus'] = [gpu.name for gpu in gpus]
    else:
        info['gpus'] = []
        
    return info