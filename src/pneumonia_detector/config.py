"""
Configuration management for pneumonia detection
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: str = "data/chest_xray"
    train_dir: str = "train"
    val_dir: str = "val" 
    test_dir: str = "test"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.2
    seed: int = 42
    
    # Data augmentation parameters
    rotation_range: int = 20
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    zoom_range: float = 0.1
    horizontal_flip: bool = True
    vertical_flip: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 2
    dropout_rate: float = 0.5
    l2_regularization: float = 0.001
    
    # Transfer learning options
    use_transfer_learning: bool = True
    base_model: str = "EfficientNetB0"  # Options: EfficientNetB0, ResNet50, VGG16
    freeze_base: bool = True
    fine_tune_at: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall"])
    
    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.2
    min_lr: float = 1e-7
    
    # Model saving
    save_best_only: bool = True
    monitor: str = "val_accuracy"
    mode: str = "max"
    



@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    experiment_name: str = "pneumonia_detection"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    log_model: bool = True
    log_artifacts: bool = True


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    
    def __post_init__(self):
        # Set derived paths
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.outputs_dir = self.project_root / "outputs"
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True) 
        self.outputs_dir.mkdir(exist_ok=True)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)