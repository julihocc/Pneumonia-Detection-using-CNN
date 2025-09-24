"""
Pneumonia Detection using CNN - A modern refactored solution
"""

__version__ = "2.0.0"
__author__ = "Pneumonia Detection Team"
__description__ = "Modern CNN-based pneumonia detection from chest X-rays"

# Import main classes for easy access
from .config import Config, DataConfig, ModelConfig, TrainingConfig, ExperimentConfig
from .data import DataPipeline, DataValidator
from .models import ModelFactory, ModelCompiler, BaseModel
from .training import Trainer, ModelEvaluator
from .inference import PneumoniaPredictor
from .utils import setup_logging, set_seed, setup_gpu, get_system_info

__all__ = [
    'Config', 'DataConfig', 'ModelConfig', 'TrainingConfig', 'ExperimentConfig',
    'DataPipeline', 'DataValidator',
    'ModelFactory', 'ModelCompiler', 'BaseModel',
    'Trainer', 'ModelEvaluator',
    'PneumoniaPredictor',
    'setup_logging', 'set_seed', 'setup_gpu', 'get_system_info'
]