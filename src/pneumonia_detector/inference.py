"""
Inference module for pneumonia detection
"""
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

from .config import Config

logger = logging.getLogger(__name__)


class PneumoniaPredictor:
    """Pneumonia prediction from chest X-ray images"""
    
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ["NORMAL", "PNEUMONIA"]
        
        logger.info(f"Loaded model from {model_path}")
        
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for prediction"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            image = image_path
            
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize to model input size
        target_size = self.config.data.image_size
        image = cv2.resize(image, target_size)
        
        # Handle grayscale vs RGB based on model input
        if self.config.model.input_shape[2] == 1:  # Grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = np.expand_dims(image, axis=-1)
        elif self.config.model.input_shape[2] == 3:  # RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_single(self, image_path: str) -> Dict:
        """Make prediction on a single image"""
        # Preprocess image
        processed_image = self._preprocess_image(image_path)
        
        # Make prediction
        prediction_proba = self.model.predict(processed_image, verbose=0)[0]
        
        # Handle binary vs multiclass
        if len(prediction_proba.shape) == 0:  # Binary classification, single output
            probability = float(prediction_proba)
            predicted_class = int(probability > 0.5)
        else:  # Multiclass or binary with 2 outputs
            probability = float(np.max(prediction_proba))
            predicted_class = int(np.argmax(prediction_proba))
            
        result = {
            'image_path': image_path,
            'prediction': self.class_names[predicted_class],
            'probability': probability,
            'confidence': probability if predicted_class == 1 else 1 - probability,
            'class_probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(prediction_proba if len(prediction_proba.shape) > 0 else [1-probability, probability])
            }
        }
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Make predictions on a batch of images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict on {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
                
        return results
    
    def predict_from_directory(self, directory_path: str, 
                             extensions: List[str] = None) -> List[Dict]:
        """Make predictions on all images in a directory"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
        directory_path = Path(directory_path)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(directory_path.glob(f"*{ext}"))
            image_paths.extend(directory_path.glob(f"*{ext.upper()}"))
            
        image_paths = [str(path) for path in image_paths]
        
        logger.info(f"Found {len(image_paths)} images in {directory_path}")
        
        return self.predict_batch(image_paths)


class BatchPredictor:
    """Efficient batch prediction for large datasets"""
    
    def __init__(self, model_path: str, config: Config, batch_size: int = 32):
        self.config = config
        self.model = tf.keras.models.load_model(model_path)
        self.batch_size = batch_size
        self.class_names = ["NORMAL", "PNEUMONIA"]
        
    def _create_dataset(self, image_paths: List[str]) -> tf.data.Dataset:
        """Create TensorFlow dataset from image paths"""
        def preprocess_path(path):
            # Load and preprocess image
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=self.config.model.input_shape[2])
            image = tf.image.resize(image, self.config.data.image_size)
            image = tf.cast(image, tf.float32) / 255.0
            return image, path
            
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def predict_batch_efficient(self, image_paths: List[str]) -> List[Dict]:
        """Efficient batch prediction using tf.data"""
        dataset = self._create_dataset(image_paths)
        
        results = []
        
        for batch_images, batch_paths in dataset:
            # Make predictions
            predictions = self.model.predict(batch_images, verbose=0)
            
            # Process predictions
            for i, (pred, path) in enumerate(zip(predictions, batch_paths)):
                path_str = path.numpy().decode('utf-8')
                
                # Handle binary vs multiclass
                if len(pred.shape) == 0:  # Binary classification, single output
                    probability = float(pred)
                    predicted_class = int(probability > 0.5)
                else:  # Multiclass or binary with 2 outputs
                    probability = float(np.max(pred))
                    predicted_class = int(np.argmax(pred))
                    
                result = {
                    'image_path': path_str,
                    'prediction': self.class_names[predicted_class],
                    'probability': probability,
                    'confidence': probability if predicted_class == 1 else 1 - probability,
                }
                
                results.append(result)
                
        return results


class ModelExplainer:
    """Model explanation and visualization tools"""
    
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.model = tf.keras.models.load_model(model_path)
        
    def generate_grad_cam(self, image_path: str, layer_name: str = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for model interpretation"""
        # This is a simplified version - full implementation would require more setup
        predictor = PneumoniaPredictor(self.model, self.config)
        processed_image = predictor._preprocess_image(image_path)
        
        # Get the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
                    
        # Create gradient model
        grad_model = tf.keras.models.Model(
            self.model.inputs,
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed_image)
            loss = predictions[:, 0]  # Assuming binary classification
            
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()