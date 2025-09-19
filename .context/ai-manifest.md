# AI Agent Manifest - Pneumonia Detection System v2.0.0

## Project Identity
- **Name**: Pneumonia Detection using CNN
- **Version**: 2.0.0
- **Type**: Medical AI / Computer Vision / MLOps
- **Domain**: Healthcare, Radiology, Deep Learning

## Core Capabilities

### Primary Function
Automated detection of pneumonia from chest X-ray images using convolutional neural networks, providing binary classification (NORMAL/PNEUMONIA) with high accuracy.

### Key Features
- **Transfer Learning**: EfficientNetB0, ResNet50, VGG16, DenseNet121 support
- **Custom CNN**: Modernized architecture from legacy implementation
- **MLOps Pipeline**: Complete experiment tracking with MLflow
- **REST API**: FastAPI-based model serving with batch prediction
- **CLI Tools**: Command-line interface for training, inference, evaluation
- **Docker Support**: Containerized deployment with development/production stacks
- **Configuration Management**: YAML-based parameter management

## Technical Architecture

### Components
1. **Configuration Layer** (`config.py`): Dataclass-based settings management
2. **Data Pipeline** (`data.py`): tf.data.Dataset with augmentation and preprocessing
3. **Model Factory** (`models.py`): Transfer learning and custom CNN creation
4. **Training System** (`training.py`): MLflow-integrated training pipeline
5. **API Server** (`api.py`): FastAPI endpoints for model serving
6. **CLI Interface** (`cli.py`): Command-line tools and utilities

### Data Flow
```
Raw X-rays → Preprocessing → Model → Prediction → API Response
     ↓              ↓            ↓         ↓           ↓
  Validation → Augmentation → Training → Evaluation → Serving
```

## Development Guidelines

### Code Standards
- **Python**: 3.8+ with type hints and dataclasses
- **Testing**: pytest with comprehensive unit/integration tests
- **Linting**: black, flake8, mypy for code quality
- **Documentation**: Sphinx for API docs, MkDocs for user guides

### Version Control
- **Branching**: main for releases, feature branches for development
- **Commits**: Descriptive messages with conventional format
- **Releases**: Semantic versioning with GitHub releases

### Deployment Strategy
- **Development**: Docker Compose with hot reload
- **Production**: Multi-stage Docker builds with nginx
- **Scaling**: Kubernetes manifests for cloud deployment

## AI Agent Instructions

### When Working on This Project
1. **Always load configuration first**: `config = Config.from_yaml("configs/default.yaml")`
2. **Use modern patterns**: tf.data.Dataset over manual loading
3. **Follow class conventions**: NORMAL=0, PNEUMONIA=1 (not legacy reverse)
4. **Test thoroughly**: Run full test suite before commits
5. **Document changes**: Update relevant docs for any modifications

### Critical Reminders
- **Data structure**: `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`
- **Model saving**: Use `config.models_dir` with timestamps
- **API limits**: Batch prediction capped at 10 images
- **MLflow**: Auto-tracking when `tracking_uri` configured

## Quality Assurance

### Testing Strategy
- **Unit Tests**: Core functions and utilities
- **Integration Tests**: End-to-end pipelines
- **Performance Tests**: Model inference speed and accuracy
- **API Tests**: Endpoint functionality and error handling

### Validation Metrics
- **Accuracy**: Target >92% (matching legacy performance)
- **Precision/Recall**: Balanced for medical application
- **Inference Speed**: <200ms per image
- **Memory Usage**: Efficient GPU utilization

## Ethical Considerations

### Medical AI Guidelines
- **Bias Awareness**: Monitor for demographic biases in training data
- **Explainability**: Grad-CAM support for prediction explanations
- **Safety**: Fail-safe error handling in production
- **Compliance**: HIPAA considerations for medical data

### Responsible Development
- **Data Privacy**: No PII in training data
- **Model Transparency**: Open-source implementation
- **Error Communication**: Clear uncertainty quantification
- **Human Oversight**: AI as decision support, not replacement

## Future Roadmap

### Planned Enhancements
- **Multi-class Classification**: COVID-19, pneumonia, normal, other
- **Model Interpretability**: SHAP values and attention mechanisms
- **Edge Deployment**: TensorFlow Lite for mobile devices
- **Continuous Learning**: Online learning with data drift detection
- **Federated Learning**: Privacy-preserving distributed training

### Research Directions
- **Uncertainty Quantification**: Bayesian neural networks
- **Few-shot Learning**: Adaptation to new hospitals/domains
- **Self-supervised Pretraining**: Unlabeled data utilization
- **Multi-modal Fusion**: Combining X-rays with clinical data

## Support and Maintenance

### Contact Information
- **Repository**: https://github.com/julihocc/Pneumonia-Detection-using-CNN
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for community support

### Maintenance Schedule
- **Security Updates**: Monthly dependency updates
- **Performance Monitoring**: Continuous model validation
- **Documentation**: Updated with each release
- **Community**: Active engagement and contribution review

---

*This manifest ensures consistent AI agent behavior and project understanding across all development and maintenance activities.*