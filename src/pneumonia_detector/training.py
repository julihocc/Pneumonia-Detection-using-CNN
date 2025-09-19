"""
Modern training pipeline with MLOps best practices
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
import numpy as np

from .config import Config
from .data import DataPipeline
from .models import ModelFactory, ModelCompiler

logger = logging.getLogger(__name__)


class Trainer:
    """Modern trainer with experiment tracking and MLOps practices"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.history = None
        self.data_pipeline = DataPipeline(config.data)
        
        # Set up MLflow
        if config.experiment.tracking_uri:
            mlflow.set_tracking_uri(config.experiment.tracking_uri)
        mlflow.set_experiment(config.experiment.experiment_name)
        
    def _create_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.config.models_dir / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=self.config.training.monitor,
                mode=self.config.training.mode,
                save_best_only=self.config.training.save_best_only,
                save_weights_only=False,
                verbose=1
            )
        )
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=self.config.training.monitor,
                mode=self.config.training.mode,
                patience=self.config.training.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Reduce learning rate on plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=self.config.training.monitor,
                mode=self.config.training.mode,
                factor=self.config.training.reduce_lr_factor,
                patience=self.config.training.reduce_lr_patience,
                min_lr=self.config.training.min_lr,
                verbose=1
            )
        )
        
        # TensorBoard logging
        log_dir = self.config.logs_dir / f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
        )
        
        # MLflow logging callback
        callbacks.append(
            MLflowLoggingCallback()
        )
        
        return callbacks
    
    def prepare_data(self, data_root: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Prepare datasets for training"""
        logger.info("Preparing datasets...")
        
        # Validate data structure
        from .data import DataValidator
        validator = DataValidator(self.config.data)
        
        if not validator.validate_data_structure(data_root):
            raise ValueError("Data validation failed")
            
        # Get dataset statistics
        stats = validator.get_dataset_statistics(data_root)
        logger.info(f"Dataset statistics: {stats}")
        
        # Create datasets
        train_ds, val_ds, test_ds = self.data_pipeline.create_datasets(data_root)
        
        return train_ds, val_ds, test_ds
    
    def build_and_compile_model(self) -> keras.Model:
        """Build and compile the model"""
        logger.info("Building model...")
        
        # Create model
        self.model = ModelFactory.create_model(self.config.model)
        
        # Compile model
        self.model = ModelCompiler.compile_model(self.model, self.config)
        
        logger.info("Model compiled successfully")
        return self.model
    
    def train(self, data_root: str, resume_from_checkpoint: Optional[str] = None) -> keras.Model:
        """Train the model"""
        with mlflow.start_run(run_name=self.config.experiment.run_name) as run:
            # Log configuration
            self._log_config()
            
            # Prepare data
            train_ds, val_ds, test_ds = self.prepare_data(data_root)
            
            # Build model
            if resume_from_checkpoint:
                logger.info(f"Loading model from checkpoint: {resume_from_checkpoint}")
                self.model = keras.models.load_model(resume_from_checkpoint)
            else:
                self.model = self.build_and_compile_model()
            
            # Log model summary
            model_summary = self._get_model_summary()
            mlflow.log_text(model_summary, "model_summary.txt")
            
            # Compute class weights
            class_weights = self.data_pipeline.compute_class_weights(data_root)
            
            # Create callbacks
            callbacks = self._create_callbacks()
            
            # Train model
            logger.info("Starting training...")
            self.history = self.model.fit(
                train_ds,
                epochs=self.config.training.epochs,
                validation_data=val_ds,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_results = self.model.evaluate(test_ds, verbose=1)
            
            # Log test results
            test_metrics = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                test_metrics[f"test_{metric_name}"] = test_results[i]
                mlflow.log_metric(f"test_{metric_name}", test_results[i])
            
            logger.info(f"Test results: {test_metrics}")
            
            # Save final model
            model_path = self.config.models_dir / f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            self.model.save(str(model_path))
            
            # Log model to MLflow
            if self.config.experiment.log_model:
                mlflow.tensorflow.log_model(self.model, "model")
            
            return self.model
    
    def _log_config(self):
        """Log configuration to MLflow"""
        # Log parameters
        mlflow.log_params({
            "batch_size": self.config.data.batch_size,
            "image_size": self.config.data.image_size,
            "epochs": self.config.training.epochs,
            "learning_rate": self.config.training.learning_rate,
            "optimizer": self.config.training.optimizer,
            "dropout_rate": self.config.model.dropout_rate,
            "l2_regularization": self.config.model.l2_regularization,
            "use_transfer_learning": self.config.model.use_transfer_learning,
            "base_model": self.config.model.base_model if self.config.model.use_transfer_learning else None,
        })
        
        # Log configuration file
        config_path = "/tmp/config.yaml"
        self.config.to_yaml(config_path)
        mlflow.log_artifact(config_path, "config")
        
    def _get_model_summary(self) -> str:
        """Get model summary as string"""
        if self.model is None:
            return "Model not built yet"
            
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        summary = buffer.getvalue()
        sys.stdout = old_stdout
        
        return summary


class MLflowLoggingCallback(keras.callbacks.Callback):
    """Custom callback for logging metrics to MLflow"""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch"""
        if logs:
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)


class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self, model: keras.Model, config: Config):
        self.model = model
        self.config = config
    
    def evaluate_comprehensive(self, test_ds: tf.data.Dataset) -> Dict:
        """Comprehensive model evaluation"""
        logger.info("Running comprehensive evaluation...")
        
        # Get predictions
        y_true = []
        y_pred_proba = []
        
        for batch_x, batch_y in test_ds:
            y_true.extend(batch_y.numpy())
            y_pred_proba.extend(self.model.predict(batch_x, verbose=0))
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Convert probabilities to predictions
        if self.config.model.num_classes == 2:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Add AUC for binary classification
        if self.config.model.num_classes == 2:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'])
        
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: keras.callbacks.History, save_path: Optional[str] = None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()