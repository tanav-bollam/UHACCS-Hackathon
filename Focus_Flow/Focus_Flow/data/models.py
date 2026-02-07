"""
Data models: Pydantic schemas for sessions, tasks, and study schedules.

Used for request/response validation and serialization across the API.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single task within a study schedule."""

    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name or description")
    duration_minutes: int = Field(..., ge=1, description="Planned duration in minutes")
    completed: bool = Field(default=False, description="Whether the task is done")


class StudySchedule(BaseModel):
    """Study schedule containing multiple tasks."""

    id: str = Field(..., description="Unique schedule identifier")
    tasks: list[Task] = Field(default_factory=list, description="List of study tasks")
    intervals: Optional[list[int]] = Field(
        default=None,
        description="Break intervals in minutes (TODO: implement)",
    )
    recurrence: Optional[str] = Field(
        default=None,
        description="Recurrence pattern e.g. daily, weekly (TODO: implement)",
    )


class Session(BaseModel):
    """A single study session record."""

    id: str = Field(..., description="Unique session identifier")
    start_time: Optional[datetime] = Field(default=None, description="Session start")
    end_time: Optional[datetime] = Field(default=None, description="Session end")
    duration_minutes: float = Field(default=0, description="Total duration in minutes")
    productivity_score: float = Field(
        default=0,
        ge=0,
        le=1,
        description="Productivity ratio 0.0-1.0",
    )
    productive_seconds: float = Field(
        default=0,
        description="Seconds counted as productive (attentive)",
    )


# --- Request/Response schemas for API endpoints ---


class SessionStartResponse(BaseModel):
    """Response when starting a new session."""

    session_id: str
    message: str = "Session started"


class SessionStopResponse(BaseModel):
    """Response when stopping a session."""

    session_id: str
    duration_minutes: float
    productivity_score: float
    message: str = "Session stopped"


class SessionSummaryResponse(BaseModel):
    """Response when fetching session details."""

    session: Session


class ScheduleRequest(BaseModel):
    """Request body for POST /schedule."""

    schedule: StudySchedule


class ScheduleResponse(BaseModel):
    """Response when saving a schedule."""

    schedule_id: str
    message: str = "Schedule saved"


class RecommendationResponse(BaseModel):
    """Response for RL tutor recommendation."""

    action: str = Field(..., description="Recommended action e.g. take_break, continue")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
