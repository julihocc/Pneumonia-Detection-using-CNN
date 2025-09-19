# Deployment Guide

## Overview

This guide covers various deployment options for the Pneumonia Detection API, from local development to production cloud deployments.

## Local Development

### Docker Development Environment

1. **Start the development stack:**
```bash
cd docker
docker-compose up -d
```

This provides:
- API server with hot reload: http://localhost:8000
- MLflow tracking: http://localhost:5000
- Jupyter Lab: http://localhost:8888
- PostgreSQL database: localhost:5432

2. **Stop the environment:**
```bash
docker-compose down
```

### Direct Python Execution

1. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
```

2. **Start the API server:**
```bash
uvicorn pneumonia_detector.api:app --host 0.0.0.0 --port 8000 --reload
```

## Production Deployment

### Docker Production Stack

1. **Build production image:**
```bash
docker build -f docker/Dockerfile.prod -t pneumonia-detector:latest .
```

2. **Start production stack:**
```bash
cd docker
docker-compose -f docker-compose.prod.yml up -d
```

This includes:
- Load-balanced API servers (3 replicas)
- Nginx reverse proxy
- MLflow with PostgreSQL backend
- Redis for caching
- Prometheus monitoring
- Grafana dashboards

### Cloud Platforms

#### AWS Deployment

**Option 1: ECS (Elastic Container Service)**

1. **Push image to ECR:**
```bash
# Create ECR repository
aws ecr create-repository --repository-name pneumonia-detector

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
docker tag pneumonia-detector:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/pneumonia-detector:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/pneumonia-detector:latest
```

2. **Create ECS task definition:**
```json
{
  "family": "pneumonia-detector",
  "networkMode": "awsvpc",
  "requiresCompatibles": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "pneumonia-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/pneumonia-detector:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/models/best_model.h5"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/pneumonia-detector",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Option 2: Lambda (for serverless)**

1. **Create deployment package:**
```bash
# Use AWS SAM or Serverless Framework
sam init --runtime python3.9 --app-template hello-world
```

2. **Configure Lambda function:**
```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  PneumoniaFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: lambda_handler.lambda_handler
      Runtime: python3.9
      Timeout: 30
      MemorySize: 3008
      Events:
        Api:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: ANY
```

#### Google Cloud Platform

**Option 1: Cloud Run**

1. **Deploy to Cloud Run:**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/pneumonia-detector

# Deploy to Cloud Run
gcloud run deploy pneumonia-detector \
    --image gcr.io/PROJECT-ID/pneumonia-detector \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

**Option 2: GKE (Google Kubernetes Engine)**

1. **Create Kubernetes manifests:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pneumonia-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pneumonia-detector
  template:
    metadata:
      labels:
        app: pneumonia-detector
    spec:
      containers:
      - name: api
        image: gcr.io/PROJECT-ID/pneumonia-detector:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: MODEL_PATH
          value: "/app/models/best_model.h5"
---
apiVersion: v1
kind: Service
metadata:
  name: pneumonia-detector-service
spec:
  selector:
    app: pneumonia-detector
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

2. **Deploy to GKE:**
```bash
kubectl apply -f k8s/deployment.yaml
```

#### Microsoft Azure

**Option 1: Container Instances**

```bash
az container create \
    --resource-group myResourceGroup \
    --name pneumonia-detector \
    --image pneumonia-detector:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8000 \
    --environment-variables MODEL_PATH=/app/models/best_model.h5
```

**Option 2: AKS (Azure Kubernetes Service)**

```bash
# Create AKS cluster
az aks create \
    --resource-group myResourceGroup \
    --name myAKSCluster \
    --node-count 3 \
    --enable-addons monitoring \
    --generate-ssh-keys

# Deploy application
kubectl apply -f k8s/deployment.yaml
```

### Environment Variables

Configure the following environment variables:

- `MODEL_PATH`: Path to the trained model file
- `CONFIG_PATH`: Path to configuration file
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_BATCH_SIZE`: Maximum batch size for batch predictions
- `CACHE_TTL`: Cache time-to-live in seconds

### Health Checks

Configure health checks for production deployments:

**HTTP Health Check:**
- Endpoint: `GET /health`
- Expected response: 200 OK
- Timeout: 30 seconds
- Interval: 30 seconds

**Readiness Probe:**
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

**Liveness Probe:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
```

### Monitoring and Logging

#### Metrics Collection

1. **Application Metrics:**
   - Request count and latency
   - Prediction accuracy
   - Model inference time
   - Error rates

2. **System Metrics:**
   - CPU and memory usage
   - GPU utilization (if applicable)
   - Disk I/O and network traffic

#### Centralized Logging

Configure centralized logging using:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Fluentd** for log collection
- **Cloud provider logging** (CloudWatch, Stackdriver, Azure Monitor)

#### Alerting

Set up alerts for:
- High error rates (>5%)
- High response times (>1s)
- Model prediction confidence drops
- System resource exhaustion

### Security Considerations

1. **API Security:**
   - Implement rate limiting
   - Add API authentication (JWT, API keys)
   - Enable HTTPS/TLS
   - Validate input files

2. **Container Security:**
   - Use non-root user
   - Scan images for vulnerabilities
   - Keep base images updated
   - Limit container capabilities

3. **Network Security:**
   - Use private networks
   - Configure firewalls
   - Enable network policies (Kubernetes)

### Scaling Strategies

1. **Horizontal Scaling:**
   - Auto-scaling based on CPU/memory
   - Load balancing across replicas
   - Queue-based processing for batch jobs

2. **Vertical Scaling:**
   - Increase CPU/memory per instance
   - Use GPU instances for faster inference
   - Optimize model size and precision

### Backup and Disaster Recovery

1. **Model Backup:**
   - Store models in object storage (S3, GCS, Azure Blob)
   - Version control for model artifacts
   - Automated backup schedules

2. **Data Backup:**
   - Database backups for user data
   - Configuration backups
   - Log retention policies

3. **Disaster Recovery:**
   - Multi-region deployments
   - Automated failover procedures
   - Recovery time objectives (RTO) planning