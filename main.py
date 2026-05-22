"""
Stock Prediction REST API
=========================
Serves next-day direction predictions for a given ticker and date range.

"""

import os
import numpy as np
from datetime import datetime, date
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import mlflow
import mlflow.pytorch
import torch
from typing import Optional

from src.data import get_inference_data
from src.data.features import return_all_features, make_transformer_sequences


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

mlflow.set_tracking_uri("file:///app/mlruns")

MLFLOW_RUN_ID = "c61919e6542a407583c9b888db098d58"
MODEL_URI = f"mlruns/0/models/m-{MLFLOW_RUN_ID}/artifacts"
WINDOW_SIZE     = int(os.getenv("WINDOW_SIZE",  "20"))
SEQ_LEN         = int(os.getenv("SEQ_LEN",      "30"))


# ─────────────────────────────────────────────────────────────────────────────
# Model state — loaded once on startup
# ─────────────────────────────────────────────────────────────────────────────

class ModelState:
    model       = None
    run_id      = None
    loaded_at   = None
    device      = None


state = ModelState()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load model on startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    try:
        print(f"Loading model from: {MODEL_URI}")       
        state.model = mlflow.pytorch.load_model(MODEL_URI, map_location=torch.device("cpu"))
        state.model.device = torch.device("cpu")
        state.device = torch.device("cpu") 
        state.model.eval()
        state.run_id    = MLFLOW_RUN_ID
        state.loaded_at = datetime.utcnow().isoformat()
        print(f"Model loaded successfully on {state.device}")
    except Exception as e:
        print(f"WARNING: Could not load model — {e}")
        print("API will start but /predict will return 503 until model is available")

    yield  # API runs here

    # Cleanup on shutdown
    state.model = None
    print("Model unloaded")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Stock Direction Prediction",
    description = "Predicts next-day price direction using a Transformer model. Simply enter the stock ticker of interest, and will return next day predicted price.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker:     str   = Field(..., example="AAPL",       description="Stock ticker symbol")

class PredictResponse(BaseModel):
    ticker:         str
    prediction:     str           # "UP" or "DOWN"
    probability:    float         # P(UP)
    predicted_at:   str


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    device:         str
    loaded_at:      Optional[str] = None
    run_id:         Optional[str] = None


class ModelInfoResponse(BaseModel):
    run_id:         Optional[str] = None
    model_uri:      str
    device:         str
    loaded_at:      Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health():
    """Check API and model status."""
    return HealthResponse(
        status       = "ok" if state.model is not None else "degraded",
        model_loaded = state.model is not None,
        device       = str(state.device) if state.device else "unknown",
        loaded_at    = state.loaded_at,
        run_id       = state.run_id,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Status"])
def model_info():
    """Return model metadata and run details."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        run_id      = state.run_id,
        model_uri   = MODEL_URI,
        device      = str(state.device),
        loaded_at   = state.loaded_at,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict next-day price direction for a given ticker.

    Returns probability of UP move.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check /health")

    # ── Load and process data via existing pipeline ───────────────────────
    today = date.today().strftime("%Y-%m-%d")
    try:
        data = get_inference_data(
            ticker         = request.ticker,
            start           = "2025-01-01",
            end             = today,
            feature_fn      = return_all_features
        )
    except Exception as e:
        import traceback
        traceback.print_exc()   # prints full traceback to terminal
        raise HTTPException(status_code=404, detail=f"Could not load data for {request.ticker}: {e}")

    if len(data) == 0:
        raise HTTPException(status_code=422, detail=f"No data returned for {request.ticker}")

    # ── Build transformer sequences ───────────────────────────────────────
    # Same make_transformer_sequences() call as training
    try:
        print(f"Data type: {type(data)}")
        print(f"Data shape/len: {data.shape if hasattr(data, 'shape') else len(data)}")
        print(f"Data sample: {data[:2] if hasattr(data, '__getitem__') else data}")
        X = make_transformer_sequences(data, 30) # Use seq len 30, limit required user input
    except Exception as e:
        import traceback
        traceback.print_exc()   # prints full traceback to terminal
        raise HTTPException(status_code=500, detail=f"Sequence construction failed: {e}")

    if len(X) == 0:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough data to build sequences of length {30}" # Use seq len 30, limit required user input
        )

    # ── Inference ─────────────────────────────────────────────────────────
    # Same slicing convention as training:
    #   signals:  X[:, :, 9:]      raw signal windows
    #   features: X[:, :, 0:7]     handcrafted features
    try:
        signals  = X[:, :, 9:].astype(np.float32)
        features = X[:, :, 0:7].astype(np.float32)

        state.device = torch.device("cpu")      
        state.model.device = torch.device("cpu")  

        print(f"model.device attribute: {state.model.device}")
        print(f"first param device: {next(state.model.parameters()).device}")

        pred, prob = state.model.evaluate(signals, features)

        # Take the last sequence's prediction — most recent view of the data
        avg_prob   = float(prob[-1])
        prediction = "UP" if avg_prob >= 0.5 else "DOWN"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        ticker       = request.ticker.upper(),
        prediction   = prediction,
        probability  = round(avg_prob, 4),
        predicted_at = datetime.utcnow().isoformat(),
    )