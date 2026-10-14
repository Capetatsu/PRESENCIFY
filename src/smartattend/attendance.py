"""
attendance.py – Session-based attendance logic (Phase 4/6).

Enforces:
  - One accepted PRESENT per student per session (DB + app level)
  - PAD_UNAVAILABLE state when no PAD model is running
  - Frames cannot create duplicate records
  - Confidence-based routing: REVIEW_REQUIRED for ambiguous matches
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session as DBSession

from .database import (
    AttendanceEvent, AttendanceSession, FaceTemplate, ReviewEvent,
    Student, has_accepted_attendance,
)
from .recognizer import MatchResult, MatchStatus


PAD_UNAVAILABLE = "PAD_UNAVAILABLE"


def record_attendance(
    db: DBSession,
    session_id: str,
    match: MatchResult,
    frame_index: Optional[int] = None,
    pad_status: str = PAD_UNAVAILABLE,
    cooldown_seconds: Optional[int] = 30,  # kept for API compatibility; no longer used
) -> Optional[AttendanceEvent]:
    """
    Persist one attendance decision. Enforces duplicate prevention.

    Returns the AttendanceEvent row, or None if blocked by duplicate rules.

    Rules:
      - PRESENT: allowed only once per student per session (is_accepted=True).
      - REVIEW_REQUIRED: queued once per student per session; ignored on repeat frames.
      - UNKNOWN: upserted — exactly one record per session is maintained; repeated
        observations update frame_index and timestamp in place (no new rows inserted).
      - MODEL_ERROR: stored for audit.

    Note: An UNKNOWN observation never implies student absence and does not assign
    any student identity. It records only that an unrecognised face was seen.
    """
    status = match.status.value


    # 1. Duplicate check (PRESENT only once per student per session)
    if status == MatchStatus.PRESENT.value and match.student_id:
        if has_accepted_attendance(db, match.student_id, session_id):
            return None  # already marked present — do not create duplicate

    # 2. De-duplication for REVIEW_REQUIRED (queue each student at most once per session)
    if status == MatchStatus.REVIEW_REQUIRED.value and match.student_id:
        existing_review = db.query(AttendanceEvent).filter_by(
            session_id=session_id,
            student_id=match.student_id,
            status=MatchStatus.REVIEW_REQUIRED.value,
        ).first()
        if existing_review:
            return None  # already queued for teacher review in this session

    # 3. UNKNOWN events: upsert — update existing observation record in place rather
    #    than inserting a new row per cooldown cycle. This keeps exactly one UNKNOWN
    #    record per session (updated timestamp + frame count), preventing queue growth.
    if status == MatchStatus.UNKNOWN.value:
        existing_unknown = db.query(AttendanceEvent).filter_by(
            session_id=session_id,
            status=MatchStatus.UNKNOWN.value,
        ).order_by(AttendanceEvent.recorded_at.desc()).first()
        if existing_unknown:
            # Update observation evidence in place; do not insert a new row
            existing_unknown.frame_index = frame_index  # tracks latest observed frame
            existing_unknown.recorded_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(existing_unknown)
            return existing_unknown  # return updated record; no duplicate created

    is_accepted = (status == MatchStatus.PRESENT.value and
                   match.student_id is not None)

    event = AttendanceEvent(
        student_id=match.student_id,
        session_id=session_id,
        status=status,
        similarity_score=match.similarity,
        similarity_threshold=match.threshold_accept,
        model_name=match.status.name,
        pad_status=pad_status,
        frame_index=frame_index,
        is_accepted=is_accepted,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event



def create_session(db: DBSession, name: str, notes: str = "") -> AttendanceSession:
    """Start a new named attendance session."""
    sess = AttendanceSession(name=name, notes=notes)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


def end_session(db: DBSession, session_id: str) -> None:
    """Mark a session as ended."""
    sess = db.query(AttendanceSession).filter_by(id=session_id).first()
    if sess:
        sess.is_active = False
        sess.ended_at = datetime.now(timezone.utc)
        db.commit()


def enroll_student(
    db: DBSession,
    display_name: str,
    pseudonym: str,
    consent: bool,
) -> Student:
    """Create a student record. Consent must be True to proceed."""
    if not consent:
        raise ValueError("Consent must be confirmed before enrollment.")
    student = Student(
        display_name=display_name,
        pseudonym=pseudonym,
        consent_given=True,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


def store_template(
    db: DBSession,
    student_id: str,
    embedding: list[float],
    model_name: str,
    model_checkpoint: str,
    preprocessing: str,
    quality_score: Optional[float] = None,
) -> FaceTemplate:
    """Store a face embedding for a student."""
    template = FaceTemplate.make(
        student_id=student_id,
        embedding=embedding,
        model_name=model_name,
        model_checkpoint=model_checkpoint,
        preprocessing=preprocessing,
        quality_score=quality_score,
    )
    db.add(template)
    db.commit()
    db.refresh(template)
    return template


def load_enrolled_embeddings(
    db: DBSession, model_name: str
) -> dict[str, tuple[str, list[float]]]:
    """
    Load all active templates for a given model.
    Returns {student_id: (display_name, embedding_vector)}.
    """
    import json
    rows = (
        db.query(FaceTemplate, Student)
        .join(Student, FaceTemplate.student_id == Student.id)
        .filter(Student.active == True, FaceTemplate.model_name == model_name)
        .all()
    )
    enrolled = {}
    for tmpl, student in rows:
        import numpy as np
        emb = np.array(json.loads(tmpl.embedding_json), dtype=np.float32)
        if student.id not in enrolled:
            enrolled[student.id] = (student.display_name, emb)
        # If multiple templates exist, average them
        else:
            existing_name, existing_emb = enrolled[student.id]
            enrolled[student.id] = (
                existing_name,
                (existing_emb + emb) / 2.0
            )
    # Re-normalise averaged embeddings
    import numpy as np
    for sid in enrolled:
        name, emb = enrolled[sid]
        norm = np.linalg.norm(emb)
        if norm > 0:
            enrolled[sid] = (name, emb / norm)
    return enrolled


def teacher_review(
    db: DBSession,
    event_id: int,
    corrected_status: str,
    reason: str,
    reviewer_note: str = "",
) -> ReviewEvent:
    """Record a teacher correction with full audit trail.

    The corrected_status is written back to the AttendanceEvent.status so the
    item is removed from the pending REVIEW_REQUIRED queue. The original status
    and teacher decision are both preserved in the ReviewEvent audit row.

    Accepted statuses: PRESENT, REJECTED, DISMISSED.
    - PRESENT: marks student accepted for the session (deduplicated).
    - REJECTED: explicitly confirms the face does not belong to any enrolled student.
    - DISMISSED: closes the item without a final verdict (e.g., camera artefact).
    """
    event = db.query(AttendanceEvent).filter_by(id=event_id).first()
    if not event:
        raise ValueError(f"AttendanceEvent {event_id} not found.")
    review = ReviewEvent(
        attendance_event_id=event_id,
        original_status=event.status,
        corrected_status=corrected_status,
        reason=reason,
        reviewer_note=reviewer_note,
    )
    # Update the event status so it leaves the pending queue
    if corrected_status == "PRESENT" and event.student_id:
        if not has_accepted_attendance(db, event.student_id, event.session_id):
            event.status = corrected_status
            event.is_accepted = True
    elif corrected_status in ("REJECTED", "DISMISSED"):
        # Store the teacher decision; mark not accepted so it cannot become PRESENT
        event.status = corrected_status
        event.is_accepted = False
    db.add(review)
    db.commit()
    db.refresh(review)
    return review

