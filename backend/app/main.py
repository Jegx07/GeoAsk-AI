"""GeoAsk AI — Main FastAPI Application.

Entry point for the backend server.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.config import get_settings

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("geoask")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Agentic Multimodal VLM for Remote-Sensing Image Analysis",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api", tags=["Analysis"])
app.include_router(ws_router, tags=["WebSocket"])

# Mount static directories for serving output images/thumbnails
app.mount("/static/uploads", StaticFiles(directory=str(settings.upload_path)), name="uploads")


@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("LLM Provider: %s", settings.llm_provider)
    
    # Ensure directories exist
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.reports_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
