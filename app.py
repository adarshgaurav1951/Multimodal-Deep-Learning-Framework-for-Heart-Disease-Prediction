"""
FastAPI Application Entry Point for Multimodal Heart Disease Research Framework.

Research-only: this API is not for diagnosis or clinical decision-making.
No prediction is fabricated when a trained model artifact is unavailable.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import settings
from src.database.session import init_db
from src.api.routes import api_router
from src.api.dependencies import get_model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables and seed baseline users
    init_db()
    # Check if a model is configured and load it
    ms = get_model_service()
    if ms.is_loaded():
        print(f"Loaded active model version: {ms.version} from {ms.artifact_path}")
    else:
        print("Starting in demo mode: No approved model artifact loaded. Run train.py to produce an artifact.")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description=(
        "Research prototype only. Outputs are experimental model scores, "
        "not medical diagnoses or clinical advice. Combines 12-lead ECG time-series "
        "and structured clinical variables via multi-scale 1D CNN, BiLSTM, and gated multimodal fusion."
    ),
    lifespan=lifespan
)

# CORS middleware for research clients and dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Mount static files directory if present
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
