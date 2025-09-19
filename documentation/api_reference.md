# API Reference

## Overview

The Pneumonia Detection API provides REST endpoints for analyzing chest X-ray images to detect pneumonia using deep learning models.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, consider implementing API keys or OAuth2.

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0"
}
```

### Single Image Prediction

**POST** `/predict`

Analyze a single chest X-ray image for pneumonia detection.

**Parameters:**
- `file` (form-data): Image file (JPEG, PNG, BMP, TIFF)

**Response:**
```json
{
  "image_name": "chest_xray.jpg",
  "prediction": "PNEUMONIA",
  "probability": 0.8542,
  "confidence": 0.8542,
  "class_probabilities": {
    "NORMAL": 0.1458,
    "PNEUMONIA": 0.8542
  }
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"
```

### Batch Prediction

**POST** `/predict_batch`

Analyze multiple chest X-ray images (max 10 per request).

**Parameters:**
- `files` (form-data): Array of image files

**Response:**
```json
{
  "results": [
    {
      "image_name": "xray1.jpg",
      "prediction": "PNEUMONIA",
      "probability": 0.8542,
      "confidence": 0.8542,
      "class_probabilities": {
        "NORMAL": 0.1458,
        "PNEUMONIA": 0.8542
      }
    }
  ],
  "total_images": 5,
  "successful_predictions": 5
}
```

### Model Information

**GET** `/model_info`

Get information about the loaded model.

**Response:**
```json
{
  "model_architecture": "Functional",
  "input_shape": [224, 224, 3],
  "num_classes": 2,
  "class_names": ["NORMAL", "PNEUMONIA"],
  "total_parameters": 4234567
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "File must be an image"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Prediction failed: Model error message"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

## Rate Limits

- 10 requests per second per IP address
- Burst up to 20 requests

## File Size Limits

- Maximum file size: 10MB per image
- Supported formats: JPEG, PNG, BMP, TIFF

## Response Times

- Single prediction: ~200ms average
- Batch prediction: ~50ms per image

## Interactive Documentation

Visit `/docs` for interactive Swagger UI documentation or `/redoc` for ReDoc documentation.

## Client Examples

### Python
```python
import requests

# Single prediction
with open('chest_xray.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']} ({result['confidence']:.3f})")
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction);
    console.log('Confidence:', data.confidence);
});
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"

# Model info
curl http://localhost:8000/model_info
```