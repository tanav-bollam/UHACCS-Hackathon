"""
Storage: CRUD operations for sessions and schedules.

Uses SQLite for persistence. Schema initialized on first import.
TODO: Add Supabase client option for cloud deployment.
TODO: Add connection pooling and error handling for production.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from data.models import Session, StudySchedule, Task

_DB_PATH = Path(__file__).resolve().parent.parent / "focus_tutor.db"
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            duration_minutes REAL DEFAULT 0,
            productivity_score REAL DEFAULT 0,
            productive_seconds REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS schedules (
            id TEXT PRIMARY KEY,
            tasks_json TEXT DEFAULT '[]',
            intervals_json TEXT,
            recurrence TEXT
        );
    """)
    conn.commit()


def create_session(session: Session) -> Session:
    """Create and store a new session."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (id, start_time, end_time, duration_minutes, productivity_score, productive_seconds) VALUES (?, ?, ?, ?, ?, ?)",
        (
            session.id,
            session.start_time.isoformat() if session.start_time else None,
            session.end_time.isoformat() if session.end_time else None,
            session.duration_minutes,
            session.productivity_score,
            session.productive_seconds,
        ),
    )
    conn.commit()
    return session


def get_session(session_id: str) -> Optional[Session]:
    """Fetch a session by ID."""
    conn = _get_conn()
    cur = conn.execute("SELECT id, start_time, end_time, duration_minutes, productivity_score, productive_seconds FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    if row is None:
        return None
    from datetime import datetime

    start = datetime.fromisoformat(row[1]) if row[1] else None
    end = datetime.fromisoformat(row[2]) if row[2] else None
    return Session(
        id=row[0],
        start_time=start,
        end_time=end,
        duration_minutes=row[3] or 0,
        productivity_score=row[4] or 0,
        productive_seconds=row[5] or 0,
    )


def list_sessions(limit: int = 20) -> list[Session]:
    """Fetch most recent sessions for frontend history view."""
    conn = _get_conn()
    cur = conn.execute(
        """
        SELECT id, start_time, end_time, duration_minutes, productivity_score, productive_seconds
        FROM sessions
        ORDER BY COALESCE(end_time, start_time) DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()

    from datetime import datetime

    sessions: list[Session] = []
    for row in rows:
        start = datetime.fromisoformat(row[1]) if row[1] else None
        end = datetime.fromisoformat(row[2]) if row[2] else None
        sessions.append(
            Session(
                id=row[0],
                start_time=start,
                end_time=end,
                duration_minutes=row[3] or 0,
                productivity_score=row[4] or 0,
                productive_seconds=row[5] or 0,
            )
        )
    return sessions


def update_session(session: Session) -> Session:
    """Update an existing session."""
    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET start_time=?, end_time=?, duration_minutes=?, productivity_score=?, productive_seconds=? WHERE id=?",
        (
            session.start_time.isoformat() if session.start_time else None,
            session.end_time.isoformat() if session.end_time else None,
            session.duration_minutes,
            session.productivity_score,
            session.productive_seconds,
            session.id,
        ),
    )
    conn.commit()
    return session


def save_schedule(schedule: StudySchedule) -> StudySchedule:
    """Save or update a study schedule."""
    conn = _get_conn()
    tasks_json = json.dumps([t.model_dump() for t in schedule.tasks])
    intervals_json = json.dumps(schedule.intervals) if schedule.intervals else None
    conn.execute(
        "INSERT OR REPLACE INTO schedules (id, tasks_json, intervals_json, recurrence) VALUES (?, ?, ?, ?)",
        (schedule.id, tasks_json, intervals_json, schedule.recurrence),
    )
    conn.commit()
    return schedule


def get_schedule(schedule_id: str) -> Optional[StudySchedule]:
    """Fetch a schedule by ID."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT id, tasks_json, intervals_json, recurrence FROM schedules WHERE id = ?",
        (schedule_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    tasks_data = json.loads(row[1] or "[]")
    tasks = [Task(**t) for t in tasks_data]
    intervals = json.loads(row[2]) if row[2] else None
    return StudySchedule(id=row[0], tasks=tasks, intervals=intervals, recurrence=row[3])


def init_storage() -> None:
    """Ensure DB and schema exist. Call on app startup."""
    _get_conn()
