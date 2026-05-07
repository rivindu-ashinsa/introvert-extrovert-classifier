from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import pandas as pd

from .schemas import InputData, PredictionResponse, FEATURE_NAMES
from .model_loader import load_model, get_model

app = FastAPI(title="Introvert-Extrovert Classifier API", version="1.0")

# Add permissive CORS middleware (adjust `allow_origins` for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    """Load the model once at application startup."""
    try:
        load_model()
    except Exception as exc:
        # Fail fast if model cannot be loaded
        raise RuntimeError(f"Failed to load model during startup: {exc}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )

@app.get("/health", tags=["Health"])
async def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}



@app.get("/", tags=["Root"])
@app.head("/", tags=["Root"])
async def root() -> dict:
    """Return a map of available endpoints and a short description.

    Useful for quick discovery and for automated checks.
    Supports both GET and HEAD for health checks.
    """
    return {
        "endpoints": {
            "/": "This endpoint (GET/HEAD) — endpoint map",
            "/health": "Health check (GET) — returns {'status': 'ok'}",
            "/predict": "Predict (POST) — accepts JSON matching InputData and returns prediction",
            "/docs": "Swagger UI (GET)",
            "/redoc": "ReDoc UI (GET)",
            "/openapi.json": "OpenAPI spec (GET)",
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: InputData) -> PredictionResponse:
    """Accept JSON payload, convert to pandas DataFrame, run model.predict, and return result.

    The InputData model uses placeholder feature names. Update `app/schemas.py` FEATURE_NAMES
    and fields to match the features your trained pipeline expects.
    """
    model = get_model()
    try:
        # Convert Pydantic model to DataFrame (single-row)
        df = pd.DataFrame([data.dict()])

        # Re-order or subset columns to match training features when possible
        if set(FEATURE_NAMES).issubset(set(df.columns)):
            df = df[FEATURE_NAMES]

        preds = model.predict(df)
        label = preds[0] if hasattr(preds, "__getitem__") else str(preds)
        return PredictionResponse(prediction=str(label))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
 