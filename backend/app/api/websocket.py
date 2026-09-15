"""GeoAsk AI — WebSocket Streaming.

Handles real-time streaming of execution trace events to the frontend.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections for live tracing."""

    def __init__(self) -> None:
        # Map session_id to active WebSocket
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a new connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("WebSocket connected for session: %s", session_id)

    def disconnect(self, session_id: str) -> None:
        """Remove a disconnected connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info("WebSocket disconnected for session: %s", session_id)

    async def send_event(self, session_id: str, event_type: str, message: str) -> None:
        """Send an event to a specific session."""
        websocket = self.active_connections.get(session_id)
        if websocket:
            try:
                await websocket.send_json({
                    "event_type": event_type,
                    "message": message,
                })
            except Exception as e:
                logger.warning("Failed to send WS event to %s: %s", session_id, e)
                self.disconnect(session_id)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for a specific session."""
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive, wait for client disconnect
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)
