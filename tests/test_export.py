"""
tests/test_export.py - Tests for Phase 8 export utilities.

Verifies:
  - export_predictions creates CSV with pseudonyms, no names, valid headers
  - export_summary creates CSV with session statistics
  - export_environment creates environment.txt with device & package info
  - export_model_manifest creates valid JSON
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartattend.attendance import create_session, enroll_student, record_attendance
from smartattend.database import get_session_factory, init_db
from smartattend.export import (
    EXPERIMENTS_DIR, export_environment, export_model_manifest,
    export_predictions, export_summary,
)
from smartattend.recognizer import MatchResult, MatchStatus


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "export_test.db")
    init_db(db_path)
    factory = get_session_factory(db_path)
    with factory() as session:
        # Populate dummy consented student and session
        student = enroll_student(session, "Charlie Brown", "P099", consent=True)
        sess = create_session(session, "Export Test Session")
        match = MatchResult(
            status=MatchStatus.PRESENT,
            student_id=student.id,
            student_name=student.display_name,
            similarity=0.88,
            threshold_accept=0.40,
            threshold_review=0.30,
            latency_ms=35.0,
        )
        record_attendance(session, sess.id, match, frame_index=1, pad_status="PAD_UNAVAILABLE")
        yield session


def test_export_predictions(test_db):
    out = export_predictions(test_db)
    assert out.exists(), "predictions CSV does not exist"
    df = pd.read_csv(out)
    assert len(df) == 1
    assert "pseudonym" in df.columns
    assert "display_name" not in df.columns, "display_name must NOT be exported"
    assert df.iloc[0]["pseudonym"] == "P099"
    assert df.iloc[0]["status"] == "PRESENT"
    assert df.iloc[0]["pad_status"] == "PAD_UNAVAILABLE"


def test_export_summary(test_db):
    out = export_summary(test_db)
    assert out.exists(), "summary CSV does not exist"
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df.iloc[0]["session_name"] == "Export Test Session"
    assert df.iloc[0]["present_accepted"] == 1


def test_export_manifest_and_environment():
    det_status = {"model": "MediaPipe BlazeFace", "status": "loaded"}
    rec_status = {"model": "EdgeFace XS", "status": "loaded"}
    p_env = export_environment(rec_status, det_status)
    assert p_env.exists()
    content = p_env.read_text(encoding="utf-8")
    assert "Python" in content or "python" in content

    p_man = export_model_manifest(rec_status, det_status)
    assert p_man.exists()
    data = json.loads(p_man.read_text(encoding="utf-8"))
    assert data["pad"]["ui_label"] == "PAD_UNAVAILABLE"
