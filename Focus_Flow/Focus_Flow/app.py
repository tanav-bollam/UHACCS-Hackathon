"""
FocusTutor: Live tutor application backend.

FastAPI application with REST endpoints for study sessions, schedules,
and RL-based tutor recommendations. Ties together CV, timers, RL, and storage.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
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
    update_session,
    save_schedule,
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


# --- Endpoints ---


@app.post("/session/start", response_model=SessionStartResponse)
def start_session() -> SessionStartResponse:
    """
    Start a new study session.

    Creates session record, starts SessionTimer and ProductivityTimer.
    Returns session_id for use in stop and recommendation.
    """
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
    """
    End a study session.

    Stops timers, computes productivity score, saves session to storage.
    Example: ProductivityTimer is updated with is_attentive from attention detector.
    """
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


@app.post("/schedule")
def submit_schedule(schedule: StudySchedule) -> dict:
    """
    Submit or update a study schedule.

    Saves schedule to storage. Can be used to plan tasks and break intervals.
    """
    save_schedule(schedule)
    return {"schedule_id": schedule.id, "message": "Schedule saved"}


@app.get("/recommendation")
def get_recommendation(session_id: Optional[str] = None) -> dict:
    """
    Fetch RL tutor recommendation.

    Uses StudyEnvironment.get_state() and RLAgent.select_action() to return
    a recommended action (e.g. continue, take_break).
    TODO: Use real reward from user feedback (e.g. did they take the break?).
    """
    if session_id and session_id in _active_sessions:
        entry = _active_sessions[session_id]
        prod_timer: ProductivityTimer = entry["productivity_timer"]
        sess_timer: SessionTimer = entry["session_timer"]
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


# --- Health check ---


@app.get("/")
def root() -> dict:
    """Health check / API info."""
    return {"status": "ok", "app": "FocusTutor"}
