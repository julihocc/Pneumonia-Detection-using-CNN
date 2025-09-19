"""
FastAPI REST API for pneumonia detection model serving
"""
import logging
from pathlib import Path
from typing import List, Optional
import tempfile
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .config import Config
from .inference import PneumoniaPredictor

logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionResponse(BaseModel):
    image_name: str
    prediction: str
    probability: float
    confidence: float
    class_probabilities: dict


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_images: int
    successful_predictions: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# Global variables for model
app = FastAPI(
    title="Pneumonia Detection API",
    description="REST API for chest X-ray pneumonia detection using deep learning",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
predictor: Optional[PneumoniaPredictor] = None


def get_predictor() -> PneumoniaPredictor:
    """Dependency to get model predictor"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    
    # Load model (these should be set via environment variables or config)
    model_path = os.getenv("MODEL_PATH", "models/best_model.h5")
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    
    try:
        if os.path.exists(model_path) and os.path.exists(config_path):
            config = Config.from_yaml(config_path)
            predictor = PneumoniaPredictor(model_path, config)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model or config file not found. API will be in maintenance mode.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if predictor is not None else "maintenance",
        model_loaded=predictor is not None,
        version="2.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    file: UploadFile = File(...),
    predictor_instance: PneumoniaPredictor = Depends(get_predictor)
):
    """Predict pneumonia from a single chest X-ray image"""
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Make prediction
        result = predictor_instance.predict_single(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return PredictionResponse(
            image_name=file.filename,
            prediction=result['prediction'],
            probability=result['probability'],
            confidence=result['confidence'],
            class_probabilities=result['class_probabilities']
        )
        
    except Exception as e:
        # Clean up temporary file in case of error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    predictor_instance: PneumoniaPredictor = Depends(get_predictor)
):
    """Predict pneumonia from multiple chest X-ray images"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    successful_predictions = 0
    temp_files = []
    
    try:
        # Save all uploaded files temporarily
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_files.append((tmp_file.name, file.filename))
        
        # Make predictions
        for tmp_path, original_name in temp_files:
            try:
                result = predictor_instance.predict_single(tmp_path)
                
                results.append(PredictionResponse(
                    image_name=original_name,
                    prediction=result['prediction'],
                    probability=result['probability'],
                    confidence=result['confidence'],
                    class_probabilities=result['class_probabilities']
                ))
                successful_predictions += 1
                
            except Exception as e:
                logger.error(f"Failed to predict on {original_name}: {e}")
                results.append(PredictionResponse(
                    image_name=original_name,
                    prediction="ERROR",
                    probability=0.0,
                    confidence=0.0,
                    class_probabilities={}
                ))
        
        return BatchPredictionResponse(
            results=results,
            total_images=len(files),
            successful_predictions=successful_predictions
        )
        
    finally:
        # Clean up temporary files
        for tmp_path, _ in temp_files:
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.get("/model_info")
async def get_model_info(predictor_instance: PneumoniaPredictor = Depends(get_predictor)):
    """Get information about the loaded model"""
    model_info = {
        "model_architecture": predictor_instance.model.__class__.__name__,
        "input_shape": predictor_instance.config.model.input_shape,
        "num_classes": predictor_instance.config.model.num_classes,
        "class_names": predictor_instance.class_names,
        "total_parameters": predictor_instance.model.count_params(),
    }
    
    return model_info


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Pneumonia Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


def serve(
    model_path: str = "models/best_model.h5",
    config_path: str = "configs/default.yaml",
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Serve the API"""
    # Set environment variables for model loading
    os.environ["MODEL_PATH"] = model_path
    os.environ["CONFIG_PATH"] = config_path
    
    uvicorn.run(
        "pneumonia_detector.api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    serve()