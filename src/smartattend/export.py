"""
export.py – Pseudonymous CSV export for experiment logging (Phase 8).

Exports:
  - predictions_anonymized.csv  (per-event; student_id replaced by pseudonym)
  - attendance_summary.csv      (per student/session counts)
  - environment.txt             (package versions, device, thresholds)
  - model_manifest.json         (model info, checkpoint, license)

Privacy:
  - display_name is NOT exported; only pseudonym (e.g. P001)
  - No raw frames or images
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session as DBSession

from .database import AttendanceEvent, AttendanceSession, ReviewEvent, Student


EXPERIMENTS_DIR = Path(__file__).parents[2] / "experiments"


def _ensure_dir():
    EXPERIMENTS_DIR.mkdir(exist_ok=True)


def export_predictions(db: DBSession, session_id: Optional[str] = None) -> Path:
    """
    Export anonymized predictions CSV.
    Uses pseudonym instead of display_name or student UUID.
    """
    _ensure_dir()
    query = (
        db.query(AttendanceEvent, Student)
        .outerjoin(Student, AttendanceEvent.student_id == Student.id)
    )
    if session_id:
        query = query.filter(AttendanceEvent.session_id == session_id)

    rows = []
    for event, student in query.all():
        rows.append({
            "event_id": event.id,
            "session_id": event.session_id,
            "pseudonym": student.pseudonym if student else "UNKNOWN",
            "status": event.status,
            "similarity_score": event.similarity_score,
            "similarity_threshold": event.similarity_threshold,
            "pad_status": event.pad_status,
            "is_accepted": event.is_accepted,
            "frame_index": event.frame_index,
            "recorded_at": event.recorded_at.isoformat() if event.recorded_at else None,
        })

    df = pd.DataFrame(rows)
    out = EXPERIMENTS_DIR / "predictions_anonymized.csv"
    df.to_csv(out, index=False)
    return out


def export_summary(db: DBSession) -> Path:
    """Export attendance summary per session."""
    _ensure_dir()
    sessions = db.query(AttendanceSession).all()
    rows = []
    for sess in sessions:
        events = db.query(AttendanceEvent).filter_by(session_id=sess.id).all()
        present = sum(1 for e in events if e.status == "PRESENT" and e.is_accepted)
        review  = sum(1 for e in events if e.status == "REVIEW_REQUIRED")
        unknown = sum(1 for e in events if e.status == "UNKNOWN")
        rejected= sum(1 for e in events if e.status == "REJECTED")
        rows.append({
            "session_id": sess.id,
            "session_name": sess.name,
            "started_at": sess.started_at.isoformat() if sess.started_at else None,
            "ended_at": sess.ended_at.isoformat() if sess.ended_at else None,
            "present_accepted": present,
            "review_required": review,
            "unknown": unknown,
            "rejected": rejected,
            "total_events": len(events),
        })
    df = pd.DataFrame(rows)
    out = EXPERIMENTS_DIR / "attendance_summary.csv"
    df.to_csv(out, index=False)
    return out


def export_environment(recognizer_status: dict, detector_status: dict) -> Path:
    """Capture exact environment for reproducibility."""
    _ensure_dir()
    lines = [
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
        "",
        "=== Package versions ===",
    ]
    try:
        result = subprocess.run(
            ["pip", "list", "--format=freeze"], capture_output=True, text=True
        )
        lines.append(result.stdout)
    except Exception:
        lines.append("(pip list failed)")
    lines += [
        "",
        "=== Detector ===",
        json.dumps(detector_status, indent=2),
        "",
        "=== Recognizer ===",
        json.dumps(recognizer_status, indent=2),
    ]
    out = EXPERIMENTS_DIR / "environment.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_model_manifest(
    recognizer_status: dict, detector_status: dict
) -> Path:
    """Export model manifest JSON for paper reproducibility section."""
    _ensure_dir()
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "detector": detector_status,
        "recognizer": recognizer_status,
        "pad": {
            "status": "NOT_IMPLEMENTED",
            "reason": "KBS 2025 paper has no public code. CASO-PAD (Sci. Reports 2026) weights unverified.",
            "ui_label": "PAD_UNAVAILABLE",
        },
    }
    out = EXPERIMENTS_DIR / "model_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
