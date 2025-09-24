# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modern, production-ready pneumonia detection system using deep learning to analyze chest X-ray images. The project is a complete refactor of an original monolithic script, implementing modern MLOps practices, modular architecture, and comprehensive tooling.

## Key Development Commands

### Installation & Setup
```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (if available)
pre-commit install
```

### Training
```bash
# Train with default configuration
pneumonia-train data/chest_xray --config-path configs/default.yaml

# Train with transfer learning
pneumonia-train data/chest_xray --config-path configs/transfer_learning.yaml

# Custom training with specific output directory
pneumonia-train data/chest_xray --config-path configs/custom_cnn.yaml --output-dir outputs/my_experiment --seed 42
```

### Prediction & Evaluation
```bash
# Single image prediction
pneumonia-predict path/to/xray.jpg models/best_model.h5

# Batch prediction
pneumonia-predict path/to/images/ models/best_model.h5 --output-path results.json

# Model evaluation on test set
pneumonia-evaluate data/chest_xray/test models/best_model.h5 --output-dir outputs/evaluation
```

### API Server
```bash
# Start API server
pneumonia-serve --model-path models/best_model.h5 --port 8000

# Alternative using uvicorn
uvicorn pneumonia_detector.api:app --host 0.0.0.0 --port 8000
```

### Testing & Code Quality
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=pneumonia_detector

# Run specific test file
pytest tests/test_config.py -v

# Code formatting and linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Docker Development
```bash
# Start development environment (includes MLflow and Jupyter)
cd docker
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### MLflow Experiment Tracking
```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Access at http://localhost:5000
```

## Code Architecture

### Core Package Structure (`src/pneumonia_detector/`)

- **`config.py`**: Configuration management using dataclasses and YAML files. Handles DataConfig, ModelConfig, TrainingConfig, and ExperimentConfig
- **`data.py`**: Modern tf.data pipeline for efficient data loading, preprocessing, and augmentation
- **`models.py`**: Model architectures including CustomCNN and TransferLearningModel with factory pattern
- **`training.py`**: Training pipeline with MLOps best practices, experiment tracking, and callbacks
- **`inference.py`**: Inference utilities for single/batch predictions with preprocessing
- **`api.py`**: FastAPI REST API server for model serving
- **`cli.py`**: Command-line interface using Typer for training, prediction, and evaluation
- **`utils.py`**: Utility functions for logging, GPU setup, system info, and seed setting

### Configuration System

The project uses YAML-based configuration with three main config files:
- **`configs/default.yaml`**: Standard configuration with EfficientNetB0 transfer learning
- **`configs/transfer_learning.yaml`**: Transfer learning specific configurations  
- **`configs/custom_cnn.yaml`**: Custom CNN architecture configuration

Configuration is managed through dataclasses in `config.py` with automatic YAML serialization/deserialization.

### Data Pipeline Architecture

- Uses tf.data API for efficient data loading and preprocessing
- Supports data augmentation (rotation, shifts, zoom, flips)
- Handles class imbalance with computed class weights
- Optimized with prefetching and parallel processing
- Expects data structure: `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`

### Model Architecture Options

1. **Transfer Learning (Recommended)**:
   - EfficientNetB0/B1 (best accuracy/efficiency)
   - ResNet50/101, VGG16/19, DenseNet121
   - Configurable fine-tuning and freezing

2. **Custom CNN**:
   - Modernized architecture with BatchNormalization
   - GlobalAveragePooling instead of Flatten
   - Configurable dropout and L2 regularization

### MLOps Integration

- **Experiment Tracking**: MLflow integration for metrics, parameters, and model artifacts
- **Model Versioning**: Automatic model checkpointing with timestamps
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Evaluation**: Comprehensive metrics including confusion matrix, classification report

### API Architecture

FastAPI-based REST API with endpoints for:
- `/predict`: Single image prediction
- `/batch-predict`: Batch prediction
- `/health`: Health check
- `/docs`: Automatic API documentation

## Dataset Requirements

The system expects the Kaggle Chest X-Ray Pneumonia dataset with structure:
```
data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

Download with:
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

## Development Workflow

1. **Config-First Development**: Always use YAML configs for experiments
2. **Testing**: Write tests for new components in `tests/`
3. **Logging**: Use structured logging throughout the codebase
4. **MLOps**: All training runs are tracked with MLflow
5. **API-First**: New functionality should be accessible via CLI and API
6. **Docker Support**: Ensure new features work in containerized environments