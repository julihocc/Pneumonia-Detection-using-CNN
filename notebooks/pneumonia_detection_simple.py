#!/usr/bin/env python3
"""
Pneumonia Detection using CNN - Simple & Working Solution
========================================================

This script implements a straightforward CNN model for pneumonia detection 
from chest X-ray images, achieving 92.6% accuracy.

Based on the original working implementation from:
Pneumonia_Detection_using_CNN(92_6_Accuracy).py

Usage:
    python pneumonia_detection_simple.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Parameters
labels = ["PNEUMONIA", "NORMAL"]
img_size = 150
EPOCHS = 12
BATCH_SIZE = 32

def install_kagglehub():
    """Install kagglehub if not already installed"""
    try:
        import kagglehub
        return kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system("pip install kagglehub")
        import kagglehub
        return kagglehub

def download_dataset():
    """Download the chest X-ray pneumonia dataset"""
    kagglehub = install_kagglehub()
    
    print("Downloading chest X-ray pneumonia dataset...")
    dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    data_dir = os.path.join(dataset_path, "chest_xray")
    print(f"Dataset downloaded to: {data_dir}")
    
    return {
        'train': os.path.join(data_dir, "train"),
        'val': os.path.join(data_dir, "val"),
        'test': os.path.join(data_dir, "test")
    }

def get_training_data(data_dir):
    """Load and preprocess images from directory structure"""
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        print(f"Loading {label} images from {path}...")
        
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist!")
            continue
            
        image_count = 0
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_arr is not None:
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    if resized_arr.shape == (img_size, img_size):
                        data.append([resized_arr, class_num])
                        image_count += 1
            except Exception as e:
                print(f"Error loading {img}: {e}")
        
        print(f"Loaded {image_count} {label} images")
    
    return np.array(data, dtype=object)

def prepare_data(train_data, val_data, test_data):
    """Prepare data arrays for training"""
    # Extract features and labels
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []
    
    for feature, label in train_data:
        x_train.append(feature)
        y_train.append(label)
        
    for feature, label in val_data:
        x_val.append(feature)
        y_val.append(label)
    
    for feature, label in test_data:
        x_test.append(feature)
        y_test.append(label)
    
    # Convert to numpy arrays and normalize
    x_train = np.array(x_train) / 255.0
    x_val = np.array(x_val) / 255.0
    x_test = np.array(x_test) / 255.0
    
    # Reshape for CNN
    x_train = x_train.reshape(-1, img_size, img_size, 1)
    x_val = x_val.reshape(-1, img_size, img_size, 1)
    x_test = x_test.reshape(-1, img_size, img_size, 1)
    
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_data_generator(x_train):
    """Create data generator for augmentation"""
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    
    datagen.fit(x_train)
    return datagen

def create_cnn_model():
    """Create the CNN model architecture"""
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    
    # Second convolutional block
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    
    # Third convolutional block
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    
    # Fourth convolutional block
    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    
    # Fifth convolutional block
    model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    
    # Dense layers
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def plot_training_history(history):
    """Plot training history"""
    epochs = range(len(history.history['accuracy']))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(epochs, history.history['accuracy'], 'go-', label='Training Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
    ax1.set_title('Training & Validation Accuracy')
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(epochs, history.history['loss'], 'g-o', label='Training Loss')
    ax2.plot(epochs, history.history['val_loss'], 'r-o', label='Validation Loss')
    ax2.set_title('Training & Validation Loss')
    ax2.legend()
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(model, x_test, y_test):
    """Evaluate model and show results"""
    # Test accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Predictions
    predictions = (model.predict(x_test) > 0.5).astype("int32").flatten()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Pneumonia', 'Normal']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    cm_df = pd.DataFrame(cm, index=['Pneumonia', 'Normal'], columns=['Pneumonia', 'Normal'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    print(f"\nDetailed Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    return predictions

def main():
    """Main execution function"""
    print("🫁 Pneumonia Detection using CNN")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Download dataset
    print("\n📥 Downloading dataset...")
    data_paths = download_dataset()
    
    # Load data
    print("\n📂 Loading data...")
    train_data = get_training_data(data_paths['train'])
    val_data = get_training_data(data_paths['val'])
    test_data = get_training_data(data_paths['test'])
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Prepare data
    print("\n🔄 Preparing data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data(train_data, val_data, test_data)
    
    # Create data generator
    print("\n🔀 Setting up data augmentation...")
    datagen = create_data_generator(x_train)
    
    # Create model
    print("\n🏗️ Building model...")
    model = create_cnn_model()
    print(f"Total parameters: {model.count_params():,}")
    
    # Training callbacks
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=2,
        verbose=1,
        factor=0.3,
        min_lr=0.000001
    )
    
    # Train model
    print(f"\n🚀 Training model for {EPOCHS} epochs...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=datagen.flow(x_val, y_val),
        callbacks=[learning_rate_reduction],
        verbose=1
    )
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n🔍 Evaluating model...")
    predictions = evaluate_model(model, x_test, y_test)
    
    # Save model
    print("\n💾 Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/pneumonia_cnn_model.h5')
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/training_history.csv', index=False)
    
    print("\n✅ Training completed successfully!")
    print("📁 Files saved:")
    print("   - models/pneumonia_cnn_model.h5")
    print("   - models/training_history.csv")
    print("   - training_history.png")
    print("   - confusion_matrix.png")

if __name__ == "__main__":
    main()