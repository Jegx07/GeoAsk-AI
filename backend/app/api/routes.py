"""GeoAsk AI — REST API Routes.

Defines the endpoints for file upload, analysis initiation, and report download.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.core.agent import GeoAskAgent
from app.models.schemas import AnalysisRequest, AnalysisResponse
from app.api.websocket import manager as ws_manager

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# In-memory store for session results (for MVP)
# In production, this would be a database (PostgreSQL/Redis)
_session_results: dict[str, AnalysisResponse] = {}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(
    background_tasks: BackgroundTasks,
    query: Annotated[str, Form()],
    files: list[UploadFile] = File(...),
) -> AnalysisResponse:
    """Submit a query and images for analysis."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    session_id = uuid.uuid4().hex[:16]
    logger.info("Starting analysis session %s for query: %s", session_id, query)

    # 1. Save uploaded files to disk
    saved_paths = []
    try:
        for file in files:
            file_path = settings.upload_path / f"{session_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_paths.append((str(file_path), file.filename))
    except Exception as e:
        logger.error("Failed to save uploaded files: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded files.")

    # 2. Process uploads via InputManager
    agent = GeoAskAgent()
    try:
        processed_images = []
        for path, original_name in saved_paths:
            img = await agent.input_manager.process_upload(path, original_name)
            processed_images.append(img)
    except Exception as e:
        logger.error("Input processing failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    # 3. Create AnalysisRequest
    input_mode = agent.input_manager.classify_input_mode(processed_images)
    request = AnalysisRequest(
        session_id=session_id,
        query=query,
        images=processed_images,
        input_mode=input_mode,
    )

    # 4. Define progress callback for WebSocket
    def _progress_cb(event_type: str, msg: str):
        # Fire-and-forget async task to send WS message
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(ws_manager.send_event(session_id, event_type, msg))
        except RuntimeError:
            pass

    # 5. Run Agent
    # For a real scalable app, we'd run this in a background task and return a 202 Accepted.
    # For this MVP and ease of use, we'll await it directly (assuming <30s execution).
    try:
        response = await agent.run(request, progress_callback=_progress_cb)
        _session_results[session_id] = response
        return response
    except Exception as e:
        logger.error("Agent execution failed: %s", e)
        raise HTTPException(status_code=500, detail="Analysis pipeline failed.")


@router.get("/sessions/{session_id}", response_model=AnalysisResponse)
async def get_session(session_id: str) -> AnalysisResponse:
    """Get the results of a completed analysis session."""
    if session_id not in _session_results:
        raise HTTPException(status_code=404, detail="Session not found.")
    return _session_results[session_id]


@router.get("/sessions/{session_id}/report")
async def download_report(session_id: str) -> FileResponse:
    """Download the PDF report for a session."""
    if session_id not in _session_results:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    response = _session_results[session_id]
    if not response.report_path or not Path(response.report_path).exists():
        raise HTTPException(status_code=404, detail="Report not generated for this session.")
        
    return FileResponse(
        path=response.report_path,
        filename=f"GeoAsk_Report_{session_id}.pdf",
        media_type="application/pdf",
    )


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
    }
