# Introvert-Extrovert Classifier — FastAPI Backend

A production-ready FastAPI backend for serving a pre-trained scikit-learn personality classification pipeline. The model predicts whether a person is an introvert or extrovert based on social behavior metrics.

## Model

- **File:** `model/social_model_v1.joblib`
- **Type:** Sklearn Pipeline (ColumnTransformer + VotingClassifier)
- **Input:** Raw social behavior features (5 numeric + 2 categorical)
- **Output:** Binary classification ("Introvert" or "Extrovert")

## Features

- FastAPI with auto-generated Swagger UI (`/docs`)
- `/` root endpoint returning endpoint map
- `/health` endpoint for monitoring
- `/predict` POST endpoint with Pydantic validation
- Case-insensitive categorical input ("yes"/"Yes"/"y"/"true" all accepted)
- CORS middleware enabled
- Containerized with Docker
- Production-ready error handling and logging

## Installation

### Local Setup (Development)

```bash
# Clone and navigate
cd introvert-extrovert-classifier

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# or macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### Access API Docs

- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON:** `http://127.0.0.1:8000/openapi.json`

## API Endpoints

### GET `/` 
Returns available endpoints and their descriptions.

**Response:**
```json
{
  "endpoints": {
    "/": "This endpoint (GET) — endpoint map",
    "/health": "Health check (GET) — returns {'status': 'ok'}",
    "/predict": "Predict (POST) — accepts JSON matching InputData and returns prediction",
    "/docs": "Swagger UI (GET)",
    "/redoc": "ReDoc UI (GET)",
    "/openapi.json": "OpenAPI spec (GET)"
  }
}
```

### GET `/health`
Simple health check for monitoring and load balancers.

**Response:**
```json
{
  "status": "ok"
}
```

### POST `/predict`
Classify a person as introvert or extrovert based on social behavior.

**Request Body:**
```json
{
  "Time_spent_Alone": 3.5,
  "Stage_fear": "Yes",
  "Social_event_attendance": 4.0,
  "Going_outside": 3.0,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 5.0,
  "Post_frequency": 2.0
}
```

**Response:**
```json
{
  "prediction": "Introvert"
}
```

### Request Field Details

| Field | Type | Range | Example | Notes |
|-------|------|-------|---------|-------|
| `Time_spent_Alone` | float | 0+ | 3.5 | Hours spent alone |
| `Stage_fear` | string | Yes/No/Y/N/True/False | "Yes" | Case-insensitive |
| `Social_event_attendance` | float | 0+ | 4.0 | Number of events attended |
| `Going_outside` | float | 0+ | 3.0 | Frequency of going outside |
| `Drained_after_socializing` | string | Yes/No/Y/N/True/False | "Yes" | Case-insensitive |
| `Friends_circle_size` | float | 0+ | 5.0 | Size of friend circle |
| `Post_frequency` | float | 0+ | 2.0 | Social media posting frequency |

## Docker

### Build Image

```bash
docker build -t introvert-extrovert-api:latest .
```

### Run Container

```bash
docker run -p 8000:8000 introvert-extrovert-api:latest
```

API will be available at `http://localhost:8000`

## Deployment on Render

1. **Connect Repository:**
   - Push code to GitHub
   - Link repo in Render dashboard

2. **Create Web Service:**
   - Select "Python" runtime
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

3. **Environment Variables (optional):**
   - Add `.env` file if needed (gitignored)

4. **Deploy:**
   - Render auto-builds on push
   - Monitor logs in dashboard

**Example Render Deployment URL:** `https://your-service-name.onrender.com`

## Development Notes

### Version Pinning

All dependencies are pinned to specific versions to ensure compatibility:
- `scikit-learn==1.6.1` (matches model training environment)
- `pandas`, `pydantic`, `fastapi`, `uvicorn` pinned for consistency

Changing versions may cause unpickling errors or breaking changes.

### Custom Transformers

The model uses a custom `BinaryMapper` transformer defined in `src/preprocess.py`. This is automatically loaded and injected during model initialization.

### Adding Features

- Update feature list in `app/schemas.py` (FEATURE_NAMES)
- Retrain model with new features
- Export updated `model/social_model_v1.joblib`
- Restart API

## Production Recommendations

- Restrict `CORSMiddleware.allow_origins` to specific domains
- Add authentication (JWT, API keys)
- Enable HTTPS/TLS
- Set up monitoring and alerting
- Use `gunicorn` with multiple workers for higher concurrency
- Add rate limiting for `/predict` endpoint

## Troubleshooting

**Model loading error:**
- Ensure `model/social_model_v1.joblib` exists at the correct path
- Check scikit-learn version matches training environment (1.6.1)

**Pydantic validation errors:**
- Verify all 7 fields are present in request JSON
- Use exact field names (case-sensitive)
- For categorical fields, use "Yes"/"No" or variants (case-insensitive)

**Port already in use:**
```bash
uvicorn app.main:app --port 8001 --reload
```

## License

MIT
 