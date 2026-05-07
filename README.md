# Introvert-Extrovert Classifier — FastAPI Backend

This repository contains a minimal production-style FastAPI backend for serving a pre-trained scikit-learn pipeline exported with `joblib`.

The model file expected by the API is:

- `model/social_model_v1.joblib`

Features
- FastAPI application with `/predict` POST endpoint and `/health` GET endpoint
- Input validation using Pydantic
- Model loaded once on startup
- CORS middleware
- Dockerfile for containerized deployment

Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Ensure your trained model is at `model/social_model_v1.joblib`.

3. Run locally (development):

```bash
uvicorn app.main:app --reload
```

By default the API will be served at `http://127.0.0.1:8000` and Swagger UI is available at `http://127.0.0.1:8000/docs`.

API Example

Request (JSON body) — update keys to match your actual model features:

```json
{
  "feature_1": 10,
  "feature_2": 5,
  "feature_3": "yes"
}
```

Response:

```json
{
  "prediction": "Introvert"
}
```

Notes & TODO
- Update `app/schemas.py` `FEATURE_NAMES` and the `InputData` fields to exactly match the features used during model training.
- For production, restrict `CORSMiddleware.allow_origins` and add HTTPS and authentication as needed.

Docker (build & run)

```bash
docker build -t introvert-extrovert-api:latest .
docker run -p 8000:8000 introvert-extrovert-api:latest
```

If you want to run with multiple workers in production, consider using `gunicorn` with `uvicorn.workers.UvicornWorker`.
 