"""
Modern CNN model architectures for pneumonia detection
"""
import logging
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from .config import ModelConfig

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for pneumonia detection models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def build_model(self) -> keras.Model:
        """Build the model architecture"""
        raise NotImplementedError
        
    def get_model_summary(self, model: keras.Model) -> str:
        """Get model summary as string"""
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        summary = buffer.getvalue()
        sys.stdout = old_stdout
        
        return summary


class CustomCNNModel(BaseModel):
    """Custom CNN model similar to the original but modernized"""
    
    def build_model(self) -> keras.Model:
        """Build custom CNN model"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.config.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fifth convolutional block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling instead of flatten
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(self.config.l2_regularization)),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(self.config.l2_regularization)),
            layers.Dropout(self.config.dropout_rate),
            
            # Output layer
            layers.Dense(1 if self.config.num_classes == 2 else self.config.num_classes,
                        activation='sigmoid' if self.config.num_classes == 2 else 'softmax')
        ])
        
        return model


class TransferLearningModel(BaseModel):
    """Transfer learning model using pre-trained architectures"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_model_map = {
            'EfficientNetB0': keras.applications.EfficientNetB0,
            'EfficientNetB1': keras.applications.EfficientNetB1,
            'ResNet50': keras.applications.ResNet50,
            'ResNet101': keras.applications.ResNet101,
            'VGG16': keras.applications.VGG16,
            'VGG19': keras.applications.VGG19,
            'DenseNet121': keras.applications.DenseNet121,
            'InceptionV3': keras.applications.InceptionV3,
            'MobileNetV2': keras.applications.MobileNetV2,
        }
        
    def build_model(self) -> keras.Model:
        """Build transfer learning model"""
        if self.config.base_model not in self.base_model_map:
            raise ValueError(f"Unsupported base model: {self.config.base_model}")
            
        # Get base model
        base_model_class = self.base_model_map[self.config.base_model]
        
        # Handle different input requirements
        if self.config.base_model.startswith('EfficientNet'):
            weights = 'imagenet'
            include_top = False
        else:
            weights = 'imagenet'
            include_top = False
            
        # Create base model
        base_model = base_model_class(
            weights=weights,
            include_top=include_top,
            input_shape=self.config.input_shape
        )
        
        # Freeze base model if requested
        if self.config.freeze_base:
            base_model.trainable = False
        elif self.config.fine_tune_at is not None:
            # Fine-tune from a specific layer
            base_model.trainable = True
            for layer in base_model.layers[:self.config.fine_tune_at]:
                layer.trainable = False
                
        # Add custom head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(self.config.l2_regularization)),
            layers.BatchNormalization(),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(self.config.l2_regularization)),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(1 if self.config.num_classes == 2 else self.config.num_classes,
                        activation='sigmoid' if self.config.num_classes == 2 else 'softmax')
        ])
        
        return model
    
    def unfreeze_base_model(self, model: keras.Model, layers_to_unfreeze: int = None):
        """Unfreeze base model layers for fine-tuning"""
        base_model = model.layers[0]
        base_model.trainable = True
        
        if layers_to_unfreeze is not None:
            # Only unfreeze the top layers_to_unfreeze layers
            for layer in base_model.layers[:-layers_to_unfreeze]:
                layer.trainable = False
                
        logger.info(f"Unfroze base model. Trainable layers: {sum(1 for layer in base_model.layers if layer.trainable)}")


class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_model(config: ModelConfig) -> keras.Model:
        """Create model based on configuration"""
        if config.use_transfer_learning:
            model_builder = TransferLearningModel(config)
        else:
            model_builder = CustomCNNModel(config)
            
        model = model_builder.build_model()
        
        logger.info(f"Created model with {model.count_params():,} parameters")
        logger.info(f"Trainable parameters: {sum(tf.keras.utils.count_params(w) for w in model.trainable_weights):,}")
        
        return model


class ModelCompiler:
    """Helper class for compiling models with different configurations"""
    
    @staticmethod
    def compile_model(model: keras.Model, config) -> keras.Model:
        """Compile model with specified configuration"""
        # Get optimizer
        if config.training.optimizer.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=config.training.learning_rate)
        elif config.training.optimizer.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=config.training.learning_rate, momentum=0.9)
        elif config.training.optimizer.lower() == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=config.training.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
            
        # Get loss function
        loss = config.training.loss_function
        
        # Get metrics
        metrics = []
        for metric_name in config.training.metrics:
            if metric_name.lower() == 'accuracy':
                metrics.append('accuracy')
            elif metric_name.lower() == 'precision':
                metrics.append(keras.metrics.Precision())
            elif metric_name.lower() == 'recall':
                metrics.append(keras.metrics.Recall())
            elif metric_name.lower() == 'auc':
                metrics.append(keras.metrics.AUC())
            elif metric_name.lower() == 'f1':
                metrics.append(keras.metrics.F1Score())
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model