"""
FocusTutor: Live tutor application backend.

FastAPI application with REST endpoints for study sessions, schedules,
and RL-based tutor recommendations. Ties together CV, timers, RL, and storage.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from analytics.metrics import calculate_productivity_score
from cv.attention import AttentionDetector, is_attentive
from cv.camera import CameraCapture
from data.models import (
    Session,
    SessionStartResponse,
    SessionStopResponse,
    SessionSummaryResponse,
    StudySchedule,
)
from data.storage import (
    create_session,
    get_session,
    init_storage,
    list_sessions,
    save_schedule,
    update_session,
)
from rl.agent import RLAgent
from rl.environment import StudyEnvironment
from timers.productivity import ProductivityTimer
from timers.session_timer import SessionTimer

app = FastAPI(
    title="FocusTutor API",
    description="Live tutor application for study sessions and productivity tracking",
    version="0.1.0",
)

# Per-session state: session_id -> {session_timer, productivity_timer}
_active_sessions: dict[str, dict] = {}

# RL components (shared across sessions for now)
rl_agent = RLAgent()
study_env = StudyEnvironment()

# CV components - use_stub=True so app runs without webcam (headless/servers)
camera = CameraCapture(use_stub=True)
attention_detector = AttentionDetector()

# Frontend assets directory (ignore zip files; serve only extracted frontend dir)
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

# Background task for periodic attention polling
_attention_task: Optional[asyncio.Task] = None


async def _poll_attention() -> None:
    """Every 2s, capture frame, run is_attentive, update productivity timers for active sessions."""
    while True:
        try:
            frame = camera.capture_frame()
            attentive = is_attentive(frame)
            for entry in _active_sessions.values():
                prod_timer: ProductivityTimer = entry["productivity_timer"]
                prod_timer.update(attentive)
        except Exception:
            pass
        await asyncio.sleep(2.0)


@app.on_event("startup")
async def startup() -> None:
    """Initialize storage (SQLite schema) and start attention polling task."""
    init_storage()
    global _attention_task
    _attention_task = asyncio.create_task(_poll_attention())


@app.on_event("shutdown")
async def shutdown() -> None:
    """Cancel attention task and release camera."""
    global _attention_task
    if _attention_task:
        _attention_task.cancel()
        try:
            await _attention_task
        except asyncio.CancelledError:
            pass
    camera.release()


# --- Request schemas ---


class SessionStopRequest(BaseModel):
    """Request body for POST /session/stop."""

    session_id: str = Field(..., description="ID of session to stop")


# --- Frontend page routes ---


@app.get("/", include_in_schema=False)
def home_page() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "home.html")


@app.get("/session", include_in_schema=False)
def session_page() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "session.html")


@app.get("/schedule", include_in_schema=False)
def schedule_page() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "schedule.html")


@app.get("/history", include_in_schema=False)
def history_page() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "history.html")


# --- API Endpoints ---


@app.get("/api/health")
def health() -> dict:
    """Health check / API info."""
    return {"status": "ok", "app": "FocusTutor"}


@app.post("/session/start", response_model=SessionStartResponse)
def start_session() -> SessionStartResponse:
    """Start a new study session and store timers."""
    session_id = str(uuid.uuid4())
    session_timer = SessionTimer()
    productivity_timer = ProductivityTimer()

    session_timer.start()
    productivity_timer.start()

    session = Session(
        id=session_id,
        start_time=datetime.utcnow(),
    )
    create_session(session)
    _active_sessions[session_id] = {
        "session_timer": session_timer,
        "productivity_timer": productivity_timer,
    }

    return SessionStartResponse(session_id=session_id)


@app.post("/session/stop", response_model=SessionStopResponse)
def stop_session(req: SessionStopRequest) -> SessionStopResponse:
    """End a study session and store productivity summary."""
    session_id = req.session_id
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found or already stopped")

    entry = _active_sessions.pop(session_id)
    session_timer: SessionTimer = entry["session_timer"]
    productivity_timer: ProductivityTimer = entry["productivity_timer"]

    frame = camera.capture_frame()
    attentive = is_attentive(frame)
    productivity_timer.update(attentive)

    session_timer.stop()
    productivity_timer.stop()

    total_seconds = session_timer.elapsed()
    productive_seconds = productivity_timer.productive_seconds
    score = calculate_productivity_score(productive_seconds, total_seconds)

    session = get_session(session_id)
    if session:
        session.end_time = datetime.utcnow()
        session.duration_minutes = total_seconds / 60
        session.productivity_score = score
        session.productive_seconds = productive_seconds
        update_session(session)

    return SessionStopResponse(
        session_id=session_id,
        duration_minutes=total_seconds / 60,
        productivity_score=score,
    )


@app.get("/session/{session_id}", response_model=SessionSummaryResponse)
def get_session_summary(session_id: str) -> SessionSummaryResponse:
    """Fetch session summary by ID."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionSummaryResponse(session=session)


@app.get("/sessions")
def get_recent_sessions(limit: int = 20) -> dict:
    """Get recent sessions for history dashboard."""
    safe_limit = max(1, min(limit, 100))
    return {"sessions": [item.model_dump() for item in list_sessions(safe_limit)]}


@app.post("/schedule")
def submit_schedule(schedule: StudySchedule) -> dict:
    """Submit or update a study schedule."""
    save_schedule(schedule)
    return {"schedule_id": schedule.id, "message": "Schedule saved"}


@app.get("/recommendation")
def get_recommendation(session_id: Optional[str] = None) -> dict:
    """Fetch RL tutor recommendation."""
    if session_id and session_id in _active_sessions:
        entry = _active_sessions[session_id]
        prod_timer: ProductivityTimer = entry["productivity_timer"]
        total = prod_timer.total_elapsed
        productive = prod_timer.productive_seconds
        study_env.set_state(
            attention_level=productive / total if total > 0 else 0,
            time_in_session_seconds=total,
            productive_ratio=productive / total if total > 0 else 0,
        )

    state = study_env.get_state()
    action, confidence = rl_agent.select_action(state)
    next_state, reward, _ = study_env.step(action)
    rl_agent.update(state, action, reward, next_state)
    return {"action": action, "confidence": confidence}
