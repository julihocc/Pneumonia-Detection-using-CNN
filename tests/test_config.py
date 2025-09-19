"""
Unit tests for configuration module
"""
import pytest
import tempfile
import os
from pathlib import Path

from pneumonia_detector.config import Config, DataConfig, ModelConfig, TrainingConfig


class TestDataConfig:
    """Test DataConfig class"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = DataConfig()
        
        assert config.data_dir == "data/chest_xray"
        assert config.train_dir == "train"
        assert config.val_dir == "val"
        assert config.test_dir == "test"
        assert config.image_size == (224, 224)
        assert config.batch_size == 32
        assert config.validation_split == 0.2
        assert config.seed == 42
        
    def test_custom_values(self):
        """Test custom configuration values"""
        config = DataConfig(
            image_size=(150, 150),
            batch_size=16,
            validation_split=0.3
        )
        
        assert config.image_size == (150, 150)
        assert config.batch_size == 16
        assert config.validation_split == 0.3


class TestModelConfig:
    """Test ModelConfig class"""
    
    def test_default_values(self):
        """Test default model configuration"""
        config = ModelConfig()
        
        assert config.input_shape == (224, 224, 3)
        assert config.num_classes == 2
        assert config.dropout_rate == 0.5
        assert config.use_transfer_learning == True
        assert config.base_model == "EfficientNetB0"
        
    def test_custom_values(self):
        """Test custom model configuration"""
        config = ModelConfig(
            input_shape=(150, 150, 1),
            use_transfer_learning=False,
            dropout_rate=0.3
        )
        
        assert config.input_shape == (150, 150, 1)
        assert config.use_transfer_learning == False
        assert config.dropout_rate == 0.3


class TestTrainingConfig:
    """Test TrainingConfig class"""
    
    def test_default_values(self):
        """Test default training configuration"""
        config = TrainingConfig()
        
        assert config.epochs == 50
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
        assert config.loss_function == "binary_crossentropy"
        assert config.metrics == ["accuracy", "precision", "recall"]
        
    def test_metrics_post_init(self):
        """Test that metrics are set properly in post_init"""
        config = TrainingConfig(metrics=None)
        assert config.metrics == ["accuracy", "precision", "recall"]


class TestConfig:
    """Test main Config class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = Config()
        
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        
        # Test that directories are created
        assert config.models_dir.exists()
        assert config.logs_dir.exists()
        assert config.outputs_dir.exists()
        
    def test_yaml_save_load(self):
        """Test saving and loading configuration from YAML"""
        config = Config()
        
        # Modify some values
        config.data.batch_size = 64
        config.model.dropout_rate = 0.3
        config.training.epochs = 100
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            temp_path = f.name
            
        try:
            # Load from file
            loaded_config = Config.from_yaml(temp_path)
            
            assert loaded_config.data.batch_size == 64
            assert loaded_config.model.dropout_rate == 0.3
            assert loaded_config.training.epochs == 100
            
        finally:
            os.unlink(temp_path)