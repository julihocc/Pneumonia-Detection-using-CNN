# AI Coding Agent Instructions

## Project Overview
This is a **modern pneumonia detection system** that analyzes chest X-ray images using deep learning. The codebase is a **complete refactor** from a legacy monolithic Jupyter notebook (`Pneumonia_Detection_using_CNN(92_6_Accuracy).py`) into a production-ready MLOps system.

## Core Architecture Components

### 1. Configuration-Driven Design
- **Entry Point**: `src/pneumonia_detector/config.py` defines all system behavior via dataclasses
- **Config Files**: `configs/{default,transfer_learning,custom_cnn}.yaml` control training/model parameters
- **Pattern**: Always load config first: `config = Config.from_yaml("configs/default.yaml")`
- **Key Classes**: `DataConfig`, `ModelConfig`, `TrainingConfig`, `ExperimentConfig`

### 2. Data Pipeline (`src/pneumonia_detector/data.py`)
- **Modern Pipeline**: Uses `tf.data.Dataset` instead of manual loading like legacy code
- **Class Names**: `["NORMAL", "PNEUMONIA"]` (NORMAL=0, PNEUMONIA=1) - opposite of legacy
- **Entry Point**: `DataPipeline(config.data).create_datasets(data_root)` returns train/val/test
- **Validation**: `DataValidator` checks directory structure before training
- **Critical**: Data expects directory structure: `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`

### 3. Model Architecture (`src/pneumonia_detector/models.py`)
- **Factory Pattern**: `ModelFactory.create_model(config.model)` handles both custom CNN and transfer learning
- **Transfer Learning Default**: EfficientNetB0 with frozen base layers
- **Legacy Alternative**: Custom CNN similar to original (when `use_transfer_learning: false`)
- **Compilation**: `ModelCompiler.compile_model(model, config)` applies optimizers/metrics

### 4. Training System (`src/pneumonia_detector/training.py`)
- **MLflow Integration**: Automatic experiment tracking (requires MLflow server)
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
- **Entry Point**: `Trainer(config).train(data_path)` handles full pipeline
- **Artifacts**: Saves models to `config.models_dir` with timestamps

## Critical Workflows

### Training a Model
```bash
# Using CLI (recommended)
pneumonia-train data/chest_xray --config-path configs/default.yaml

# Using Python script
python scripts/train.py data/chest_xray --config configs/transfer_learning.yaml
```

### Starting API Server
```bash
# Development
uvicorn pneumonia_detector.api:app --reload --port 8000

# Production with Docker
docker-compose -f docker/docker-compose.prod.yml up -d
```

### Docker Development Stack
```bash
cd docker && docker-compose up -d
# Provides: API (8000), MLflow (5000), Jupyter (8888), PostgreSQL (5432)
```

## Project-Specific Patterns

### 1. Configuration Inheritance
- `configs/default.yaml` contains base settings
- Other configs override specific sections (e.g., `transfer_learning.yaml` changes model settings)
- Use `Config.from_yaml()` to load, never hardcode values

### 2. Dataclass-Based Configuration
- All configs are strongly typed dataclasses with defaults
- Post-init hooks create directories automatically
- YAML serialization preserves all settings for reproducibility

### 3. Modern vs Legacy Code Split
- **Legacy**: `Pneumonia_Detection_using_CNN(92_6_Accuracy).py` (single file, manual preprocessing)
- **Modern**: `src/pneumonia_detector/` (modular, tf.data, MLOps)
- **Key Difference**: Legacy uses manual OpenCV loading; modern uses `tf.keras.utils.image_dataset_from_directory`

### 4. Class Label Convention
- **Legacy code**: PNEUMONIA=0, NORMAL=1 
- **Modern code**: NORMAL=0, PNEUMONIA=1 (standard convention)
- **Critical**: Prediction interpretation depends on this mapping

### 5. Entry Points and Commands
- **CLI**: `pneumonia-train`, `pneumonia-predict`, `pneumonia-evaluate` (defined in `setup.py`)
- **API**: FastAPI server with `/predict`, `/predict_batch`, `/health` endpoints
- **Scripts**: `scripts/train.py`, `scripts/setup_data.py` for automation

## Development Conventions

### Testing Pattern
- Use `pytest` with fixtures in `tests/conftest.py`
- Mock TensorFlow imports for unit tests (many tests skip if TF unavailable)
- Test config loading/validation extensively before model tests

### Logging Strategy
- Use `setup_logging()` from utils for consistent formatting
- MLflow automatically logs training metrics and model artifacts
- Console output uses `rich` library for formatted tables/progress

### Error Handling
- API uses HTTPException with specific status codes
- Training failures save partial progress before raising
- Data validation fails fast with descriptive messages

## Integration Points

### MLflow Experiment Tracking
- Auto-started in `Trainer.__init__()` if tracking_uri set
- Logs: hyperparameters, metrics, model artifacts, training curves
- View at http://localhost:5000 when using docker-compose

### Docker Multi-Stage Setup
- `docker/Dockerfile`: Development with hot reload
- `docker/Dockerfile.prod`: Multi-stage build for production
- `docker-compose.yml`: Full development stack with services
- `nginx.conf`: Load balancer with rate limiting for production

### Model Serving Architecture
- FastAPI app with global model instance
- Startup event loads model from environment variables
- Health checks verify model availability
- Batch prediction limited to 10 images per request

## Key Files to Understand First
1. `src/pneumonia_detector/config.py` - System behavior definition
2. `src/pneumonia_detector/training.py` - Core training logic
3. `configs/default.yaml` - Default system settings
4. `docker/docker-compose.yml` - Development environment setup
5. `src/pneumonia_detector/api.py` - Model serving implementation