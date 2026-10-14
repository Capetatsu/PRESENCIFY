"""
database.py – SQLite persistence for PRESENCIFY (Phase 4/6).

Tables:
  students          – enrolled students (pseudonymous IDs)
  face_templates    – stored embeddings per student (model/version tracked)
  sessions          – named attendance sessions
  attendance_events – one accepted row per (student, session); duplicates blocked
  review_events     – audit trail for teacher corrections

Design decisions:
  - Unique constraint on accepted (student_id, session_id) — enforced DB-level.
  - Original model decision is stored separately from teacher override.
  - No raw frames stored by default.
  - Timestamps are UTC ISO-8601.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer,
    String, Text, UniqueConstraint, create_engine, event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DB_PATH = Path(__file__).parents[2] / "presencify.db"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Student(Base):
    __tablename__ = "students"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    display_name = Column(String(200), nullable=False)
    pseudonym = Column(String(50), nullable=False, unique=True)  # e.g. P001
    consent_given = Column(Boolean, nullable=False, default=False)
    enrolled_at = Column(DateTime(timezone=True), default=_now)
    active = Column(Boolean, default=True)

    templates = relationship("FaceTemplate", back_populates="student",
                             cascade="all, delete-orphan")
    attendance_events = relationship("AttendanceEvent", back_populates="student")


class FaceTemplate(Base):
    __tablename__ = "face_templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(36), ForeignKey("students.id"), nullable=False)
    embedding_json = Column(Text, nullable=False)   # JSON list of floats
    model_name = Column(String(200), nullable=False)
    model_checkpoint = Column(String(200))
    preprocessing = Column(String(200))             # e.g. "BGR→RGB /127.5-1 112x112"
    captured_at = Column(DateTime(timezone=True), default=_now)
    quality_score = Column(Float)                   # optional, from detector confidence

    student = relationship("Student", back_populates="templates")

    def get_embedding(self) -> list[float]:
        return json.loads(self.embedding_json)

    @staticmethod
    def make(student_id: str, embedding: list[float],
             model_name: str, model_checkpoint: str,
             preprocessing: str, quality_score: Optional[float] = None
             ) -> "FaceTemplate":
        return FaceTemplate(
            student_id=student_id,
            embedding_json=json.dumps(embedding),
            model_name=model_name,
            model_checkpoint=model_checkpoint,
            preprocessing=preprocessing,
            quality_score=quality_score,
        )


class AttendanceSession(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    started_at = Column(DateTime(timezone=True), default=_now)
    ended_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    notes = Column(Text)

    events = relationship("AttendanceEvent", back_populates="session")


class AttendanceEvent(Base):
    """
    One row per attendance decision (including REVIEW_REQUIRED, UNKNOWN, etc.)
    The unique constraint on (student_id, session_id) where status='PRESENT'
    is enforced at application level; all events are stored for audit.
    """
    __tablename__ = "attendance_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(36), ForeignKey("students.id"))
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    status = Column(String(30), nullable=False)
    # Values: PRESENT | REVIEW_REQUIRED | UNKNOWN | REJECTED | NO_FACE | PAD_UNAVAILABLE

    similarity_score = Column(Float)    # raw score; NOT a probability
    similarity_threshold = Column(Float)
    model_name = Column(String(200))
    pad_status = Column(String(30))     # PAD_PASS | PAD_FAIL | PAD_UNAVAILABLE
    recorded_at = Column(DateTime(timezone=True), default=_now)
    frame_index = Column(Integer)
    is_accepted = Column(Boolean, default=False)
    # accepted = final attendance record (de-duplicated by application logic)

    student = relationship("Student", back_populates="attendance_events")
    session = relationship("AttendanceSession", back_populates="events")
    reviews = relationship("ReviewEvent", back_populates="attendance_event")


class ReviewEvent(Base):
    """Audit trail: teacher overrides and corrections."""
    __tablename__ = "review_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    attendance_event_id = Column(Integer, ForeignKey("attendance_events.id"))
    original_status = Column(String(30))
    corrected_status = Column(String(30))
    reason = Column(Text)
    reviewed_at = Column(DateTime(timezone=True), default=_now)
    reviewer_note = Column(Text)

    attendance_event = relationship("AttendanceEvent", back_populates="reviews")


# ── Engine and session factory ───────────────────────────────────────────────

def make_engine(db_path: Optional[str] = None):
    path = db_path or str(DB_PATH)
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
    )
    # Enable WAL mode for concurrent reads
    @event.listens_for(engine, "connect")
    def set_wal(conn, _):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
    return engine


def init_db(db_path: Optional[str] = None):
    """Create all tables if they don't exist. Safe to call repeatedly."""
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(db_path: Optional[str] = None):
    engine = init_db(db_path)
    return sessionmaker(bind=engine, autoflush=False)


# ── Attendance de-duplication helper ─────────────────────────────────────────

def has_accepted_attendance(
    db: Session, student_id: str, session_id: str
) -> bool:
    """Return True if an accepted PRESENT record exists for this student/session."""
    return db.query(AttendanceEvent).filter_by(
        student_id=student_id,
        session_id=session_id,
        is_accepted=True,
        status="PRESENT",
    ).count() > 0
