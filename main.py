"""
House Price Prediction - FastAPI Backend
College Project: Machine Learning with FastAPI & scikit-learn
Serves predictions, model performance metrics, and static frontend.
"""

import json
import os
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
LINEAR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_pipeline.joblib")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_pipeline.joblib")

# Initialize FastAPI application
app = FastAPI(
    title="HomeValue AI - House Price Prediction API",
    description="College ML project backend predicting house prices with Linear Regression and Random Forest models.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
models_cache = {
    "linear": None,
    "random_forest": None,
    "metrics": None
}


def load_cached_models():
    """Load trained model pipelines and metrics into memory."""
    if os.path.exists(LINEAR_MODEL_PATH):
        try:
            models_cache["linear"] = joblib.load(LINEAR_MODEL_PATH)
            print("Loaded Linear Regression pipeline successfully.")
        except Exception as e:
            print(f"Error loading Linear model: {e}")

    if os.path.exists(RF_MODEL_PATH):
        try:
            models_cache["random_forest"] = joblib.load(RF_MODEL_PATH)
            print("Loaded Random Forest pipeline successfully.")
        except Exception as e:
            print(f"Error loading Random Forest model: {e}")

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                models_cache["metrics"] = json.load(f)
            print("Loaded metrics.json successfully.")
        except Exception as e:
            print(f"Error loading metrics.json: {e}")


@app.on_event("startup")
async def startup_event():
    """Run model loading on API startup."""
    load_cached_models()


# Pydantic schema for prediction input validation
class PredictionRequest(BaseModel):
    living_area: float = Field(
        ...,
        gt=0,
        description="Above grade living area in square feet (GrLivArea). Must be > 0."
    )
    overall_quality: int = Field(
        ...,
        ge=1,
        le=10,
        description="Overall material and finish rating (OverallQual). Must be between 1 and 10."
    )
    bedrooms: int = Field(
        ...,
        ge=0,
        le=20,
        description="Bedrooms above grade (BedroomAbvGr). Must be >= 0."
    )
    bathrooms: int = Field(
        ...,
        ge=0,
        le=10,
        description="Full bathrooms above grade (FullBath). Must be >= 0."
    )
    year_built: int = Field(
        ...,
        ge=1800,
        le=2050,
        description="Original construction year (YearBuilt). Must be between 1800 and 2050."
    )
    garage_cars: int = Field(
        ...,
        ge=0,
        le=10,
        description="Garage car capacity (GarageCars). Must be >= 0."
    )
    basement_area: float = Field(
        ...,
        ge=0,
        description="Total basement area in square feet (TotalBsmtSF). Must be >= 0."
    )
    neighborhood: str = Field(
        ...,
        min_length=1,
        description="Neighborhood location code in Ames, Iowa (e.g. 'CollgCr', 'OldTown', 'NridgHt')."
    )
    model: Literal["linear", "random_forest"] = Field(
        default="random_forest",
        description="Model choice: 'linear' for Linear Regression or 'random_forest' for Random Forest."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "living_area": 1710,
                "overall_quality": 7,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 2003,
                "garage_cars": 2,
                "basement_area": 856,
                "neighborhood": "CollgCr",
                "model": "random_forest"
            }
        }


class PredictionResponse(BaseModel):
    predicted_price: float
    range_low: float
    range_high: float
    model_used: str


@app.get("/health", tags=["Status"])
async def health_check():
    """
    Health check endpoint returning system status.
    """
    return {"status": "ok"}


@app.get("/metrics", tags=["Performance"])
async def get_metrics():
    """
    Returns model performance metrics, top feature importances,
    and actual-vs-predicted test samples from metrics.json.
    """
    # If not cached, attempt to re-read
    if not models_cache["metrics"] and os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                models_cache["metrics"] = json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read metrics: {str(e)}"
            )

    if not models_cache["metrics"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found. Please train models first using 'python train.py'."
        )

    return models_cache["metrics"]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_price(request: PredictionRequest):
    """
    Predicts house sale price based on input property attributes.
    Returns the predicted price alongside an estimated ±8% valuation range.
    """
    model_key = request.model.lower()
    pipeline = models_cache.get(model_key)

    # If model is not loaded in memory, try to load it from disk
    if pipeline is None:
        model_file = LINEAR_MODEL_PATH if model_key == "linear" else RF_MODEL_PATH
        if os.path.exists(model_file):
            pipeline = joblib.load(model_file)
            models_cache[model_key] = pipeline
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{request.model}' is not trained yet. Run 'python train.py' to train it."
            )

    # Format input into a pandas DataFrame matching model training features
    input_df = pd.DataFrame([{
        "GrLivArea": float(request.living_area),
        "OverallQual": int(request.overall_quality),
        "BedroomAbvGr": int(request.bedrooms),
        "FullBath": int(request.bathrooms),
        "YearBuilt": int(request.year_built),
        "GarageCars": int(request.garage_cars),
        "TotalBsmtSF": float(request.basement_area),
        "Neighborhood": str(request.neighborhood).strip()
    }])

    try:
        # Predict using scikit-learn pipeline (handles imputation, scaling, and one-hot encoding)
        raw_prediction = pipeline.predict(input_df)[0]
        predicted_price = float(raw_prediction)

        # Floor at 0 for edge cases
        if predicted_price < 0:
            predicted_price = 0.0

        # Calculate ±8% confidence range
        range_low = round(predicted_price * 0.92, 2)
        range_high = round(predicted_price * 1.08, 2)
        predicted_price = round(predicted_price, 2)

        model_name = (
            "Linear Regression" if model_key == "linear" else "Random Forest Regressor"
        )

        return PredictionResponse(
            predicted_price=predicted_price,
            range_low=range_low,
            range_high=range_high,
            model_used=model_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


# Mount static assets directory
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def serve_index():
    """Serve the single-page frontend application."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(
        content={
            "message": "HomeValue AI API is running.",
            "docs": "/docs",
            "frontend": "index.html not found in static folder."
        }
    )
