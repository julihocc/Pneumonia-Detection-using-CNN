"""
Modern data pipeline using tf.data API for efficient data loading and preprocessing
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .config import DataConfig

logger = logging.getLogger(__name__)


class DataPipeline:
    """Modern data pipeline for pneumonia detection"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.class_names = ["NORMAL", "PNEUMONIA"]
        self.num_classes = len(self.class_names)
        
    def _get_dataset_from_directory(self, data_dir: str, subset: str = None) -> tf.data.Dataset:
        """Create dataset from directory structure"""
        return tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.config.validation_split if subset else None,
            subset=subset,
            seed=self.config.seed,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            class_names=self.class_names
        )
    
    def _preprocess_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess individual image"""
        # Normalize pixel values to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Convert to grayscale if needed (original dataset is grayscale)
        if tf.shape(image)[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
            
        # Ensure 3 channels for transfer learning models
        if self.config.image_size == (224, 224):  # Assuming transfer learning
            image = tf.image.grayscale_to_rgb(image)
            
        return image, label
    
    def _augment_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply data augmentation"""
        # Random rotation
        if self.config.rotation_range > 0:
            image = tf.image.rot90(
                image, 
                k=tf.random.uniform([], 0, 4, dtype=tf.int32)
            )
        
        # Random horizontal flip
        if self.config.horizontal_flip:
            image = tf.image.random_flip_left_right(image)
            
        # Random vertical flip  
        if self.config.vertical_flip:
            image = tf.image.random_flip_up_down(image)
            
        # Random zoom and shift (using random crop and resize)
        if self.config.zoom_range > 0:
            shape = tf.shape(image)
            height, width = shape[0], shape[1]
            
            # Calculate crop size
            crop_factor = 1.0 - self.config.zoom_range
            crop_height = tf.cast(tf.cast(height, tf.float32) * crop_factor, tf.int32)
            crop_width = tf.cast(tf.cast(width, tf.float32) * crop_factor, tf.int32)
            
            # Random crop
            image = tf.image.random_crop(image, [crop_height, crop_width, tf.shape(image)[2]])
            
            # Resize back to original size
            image = tf.image.resize(image, [height, width])
            
        return image, label
    
    def _prepare_dataset(self, dataset: tf.data.Dataset, 
                        augment: bool = False, 
                        cache: bool = True) -> tf.data.Dataset:
        """Prepare dataset with preprocessing and optimization"""
        # Preprocess images
        dataset = dataset.map(
            self._preprocess_image, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation if requested
        if augment:
            dataset = dataset.map(
                self._augment_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Cache dataset if requested
        if cache:
            dataset = dataset.cache()
            
        # Shuffle and prefetch for performance
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    def create_datasets(self, data_root: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create train, validation, and test datasets"""
        data_root = Path(data_root)
        
        # Load datasets
        train_dir = data_root / self.config.train_dir
        val_dir = data_root / self.config.val_dir
        test_dir = data_root / self.config.test_dir
        
        logger.info(f"Loading training data from {train_dir}")
        train_ds = self._get_dataset_from_directory(str(train_dir))
        
        logger.info(f"Loading validation data from {val_dir}")
        val_ds = self._get_dataset_from_directory(str(val_dir))
        
        logger.info(f"Loading test data from {test_dir}")
        test_ds = self._get_dataset_from_directory(str(test_dir))
        
        # Prepare datasets
        train_ds = self._prepare_dataset(train_ds, augment=True)
        val_ds = self._prepare_dataset(val_ds, augment=False)
        test_ds = self._prepare_dataset(test_ds, augment=False)
        
        return train_ds, val_ds, test_ds
    
    def compute_class_weights(self, data_root: str) -> Dict[int, float]:
        """Compute class weights for imbalanced dataset"""
        data_root = Path(data_root)
        train_dir = data_root / self.config.train_dir
        
        # Count samples in each class
        class_counts = {}
        for i, class_name in enumerate(self.class_names):
            class_dir = train_dir / class_name
            if class_dir.exists():
                class_counts[i] = len(list(class_dir.glob("*")))
            else:
                logger.warning(f"Class directory {class_dir} not found")
                class_counts[i] = 0
        
        # Compute class weights
        total_samples = sum(class_counts.values())
        class_weights = {}
        
        for class_idx, count in class_counts.items():
            if count > 0:
                class_weights[class_idx] = total_samples / (len(class_counts) * count)
            else:
                class_weights[class_idx] = 1.0
                
        logger.info(f"Class weights: {class_weights}")
        return class_weights


class DataValidator:
    """Validate data quality and structure"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def validate_data_structure(self, data_root: str) -> bool:
        """Validate that data directory has correct structure"""
        data_root = Path(data_root)
        
        required_dirs = [
            data_root / self.config.train_dir,
            data_root / self.config.val_dir,
            data_root / self.config.test_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Required directory {dir_path} does not exist")
                return False
                
            # Check for class subdirectories
            class_dirs = [dir_path / "NORMAL", dir_path / "PNEUMONIA"]
            for class_dir in class_dirs:
                if not class_dir.exists():
                    logger.error(f"Class directory {class_dir} does not exist")
                    return False
                    
                # Check if directory has images
                image_files = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                if len(image_files) == 0:
                    logger.warning(f"No image files found in {class_dir}")
                    
        logger.info("Data structure validation passed")
        return True
    
    def get_dataset_statistics(self, data_root: str) -> Dict:
        """Get statistics about the dataset"""
        data_root = Path(data_root)
        stats = {}
        
        for split in [self.config.train_dir, self.config.val_dir, self.config.test_dir]:
            split_dir = data_root / split
            split_stats = {"total": 0, "classes": {}}
            
            for class_name in ["NORMAL", "PNEUMONIA"]:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob("*")))
                    split_stats["classes"][class_name] = count
                    split_stats["total"] += count
                else:
                    split_stats["classes"][class_name] = 0
                    
            stats[split] = split_stats
            
        return stats