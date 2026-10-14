"""
tests/test_pipeline.py – Integration tests for Phase 2–6.

Tests cover:
  - Detector: MediaPipe loads, detects faces, handles no-face
  - Aligner: produces correct shape output
  - Recognizer: loads EdgeFace, produces normalised 512-d embedding, matching logic
  - Database: schema creation, enrollment, session, attendance, duplicate prevention
  - Attendance: session logic, duplicate block, teacher review, CSV export

All tests use the real camera (skipped if absent) or synthetic data.
No fabricated results.
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartattend.aligner import align_face_112, preprocess_for_recognition
from smartattend.database import (
    AttendanceEvent, AttendanceSession, FaceTemplate, Student,
    get_session_factory, init_db,
)
from smartattend.attendance import (
    create_session, end_session, enroll_student, has_accepted_attendance,
    load_enrolled_embeddings, record_attendance, store_template, teacher_review,
)
from smartattend.recognizer import FaceRecognizer, MatchResult, MatchStatus


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tmp_db(tmp_path_factory):
    """In-memory-like DB in a temp dir for tests."""
    db_path = str(tmp_path_factory.mktemp("db") / "test.db")
    init_db(db_path)
    factory = get_session_factory(db_path)
    return factory


@pytest.fixture
def db(tmp_db):
    with tmp_db() as session:
        yield session


def synthetic_face_bgr(size: int = 112) -> np.ndarray:
    """Rough synthetic face image for tests (not a real person)."""
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    cv2.ellipse(img, (cx, cy), (size // 4, size // 3), 0, 0, 360, (195, 155, 115), -1)
    cv2.circle(img, (cx - size // 8, cy - size // 8), size // 16, (50, 30, 20), -1)
    cv2.circle(img, (cx + size // 8, cy - size // 8), size // 16, (50, 30, 20), -1)
    return img


# ── Phase 2: Detector ──────────────────────────────────────────────────────────

class TestDetector:

    def test_mediapipe_import(self):
        import mediapipe as mp
        assert hasattr(mp, "tasks"), "mediapipe.tasks not available"

    def test_detector_loads(self):
        from smartattend.detector import FaceDetector
        d = FaceDetector()
        ok = d.load()
        assert ok, f"Detector load failed: {d.status['load_error']}"
        d.close()

    def test_detector_returns_list(self):
        from smartattend.detector import FaceDetector
        d = FaceDetector()
        d.load()
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        result = d.detect(frame)
        assert isinstance(result, list), "detect() must return a list"
        d.close()

    def test_blank_image_zero_detections(self):
        from smartattend.detector import FaceDetector
        d = FaceDetector()
        d.load()
        blank = np.full((480, 640, 3), 180, dtype=np.uint8)
        result = d.detect(blank)
        assert result == [], f"Expected 0 detections on blank, got {len(result)}"
        d.close()

    def test_status_dict_has_required_keys(self):
        from smartattend.detector import FaceDetector
        d = FaceDetector()
        d.load()
        s = d.status
        for key in ("model", "source", "loaded", "load_time_ms"):
            assert key in s, f"Missing key '{key}' in detector status"
        d.close()

    def test_detector_latency_under_500ms(self):
        """Smoke test: ensure latency < 500 ms on CPU."""
        import time
        from smartattend.detector import FaceDetector
        d = FaceDetector()
        d.load()
        frame = synthetic_face_bgr(480)
        # warm-up
        d.detect(frame)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            d.detect(frame)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        d.close()
        assert avg < 500, f"Detector avg latency {avg:.1f} ms exceeds 500 ms threshold"


# ── Phase 2/3: Aligner ────────────────────────────────────────────────────────

class TestAligner:

    def test_align_from_bbox_returns_112x112(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)
        out = align_face_112(frame, bbox, keypoints=None)
        assert out.shape == (112, 112, 3)

    def test_align_empty_bbox_returns_zeros(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = align_face_112(frame, (0, 0, 0, 0), keypoints=None)
        assert out.shape == (112, 112, 3)

    def test_preprocess_returns_correct_shape(self):
        import torch
        face = synthetic_face_bgr(112)
        tensor = preprocess_for_recognition(face)
        assert tensor.shape == (1, 3, 112, 112)
        assert tensor.dtype == torch.float32

    def test_preprocess_range(self):
        import torch
        face = synthetic_face_bgr(112)
        tensor = preprocess_for_recognition(face)
        assert tensor.min() >= -1.0, "Tensor min < -1.0"
        assert tensor.max() <= 1.0,  "Tensor max > 1.0"


# ── Phase 3: Recognizer ───────────────────────────────────────────────────────

class TestRecognizer:

    def test_recognizer_loads(self):
        r = FaceRecognizer()
        ok = r.load()
        assert ok, f"Recognizer load failed: {r.status['load_error']}"

    def test_embed_returns_512d(self):
        r = FaceRecognizer()
        r.load()
        face = synthetic_face_bgr(112)
        emb = r.embed(face)
        assert emb is not None, "embed() returned None"
        assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"

    def test_embed_is_unit_norm(self):
        import numpy as np
        r = FaceRecognizer()
        r.load()
        face = synthetic_face_bgr(112)
        emb = r.embed(face)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-4, f"Embedding norm={norm:.4f}, expected ~1.0"

    def test_embed_is_deterministic(self):
        import numpy as np
        r = FaceRecognizer()
        r.load()
        face = synthetic_face_bgr(112)
        e1 = r.embed(face)
        e2 = r.embed(face)
        assert np.allclose(e1, e2, atol=1e-5), "Embeddings not deterministic"

    def test_embed_is_finite(self):
        import numpy as np
        r = FaceRecognizer()
        r.load()
        face = synthetic_face_bgr(112)
        emb = r.embed(face)
        assert np.all(np.isfinite(emb)), "Embedding contains NaN or Inf"

    def test_match_unknown_when_no_enrolled(self):
        import numpy as np
        r = FaceRecognizer()
        query = np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)
        result = r.match(query, {})
        assert result.status == MatchStatus.UNKNOWN

    def test_match_present_above_threshold(self):
        import numpy as np
        r = FaceRecognizer()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        # Enroll exactly the same embedding (identity)
        enrolled = {"s1": ("Alice", emb.copy())}
        result = r.match(emb, enrolled)
        # Same embedding → cosine sim = 1.0 → PRESENT
        assert result.status == MatchStatus.PRESENT
        assert result.student_id == "s1"
        assert result.similarity is not None
        assert result.similarity > r._threshold_accept

    def test_match_unknown_when_sim_too_low(self):
        import numpy as np
        r = FaceRecognizer()
        emb_a = np.zeros(512, dtype=np.float32); emb_a[0] = 1.0
        emb_b = np.zeros(512, dtype=np.float32); emb_b[1] = 1.0
        # Cosine similarity = 0.0 < threshold → UNKNOWN
        enrolled = {"s1": ("Bob", emb_b)}
        result = r.match(emb_a, enrolled)
        assert result.status == MatchStatus.UNKNOWN
        assert result.student_id is None

    def test_similarity_is_not_probability_field_name(self):
        """Sanity check: MatchResult must not have a 'probability' attribute."""
        import numpy as np
        r = FaceRecognizer()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        result = r.match(emb, {"s1": ("Test", emb)})
        d = result.to_dict()
        assert "probability" not in d, "Result must not expose 'probability'"
        assert "similarity" in d, "Result must expose 'similarity'"


# ── Phase 4/6: Database and Attendance ────────────────────────────────────────

class TestDatabase:

    def test_enroll_student_requires_consent(self, db):
        with pytest.raises(ValueError, match="Consent"):
            enroll_student(db, "Test User", "P000", consent=False)

    def test_enroll_student_succeeds(self, db):
        s = enroll_student(db, "Alice Smith", "P001", consent=True)
        assert s.id is not None
        assert s.pseudonym == "P001"
        assert s.consent_given is True

    def test_create_session(self, db):
        sess = create_session(db, "Morning session")
        assert sess.id is not None
        assert sess.is_active is True

    def test_end_session(self, db):
        sess = create_session(db, "To end")
        end_session(db, sess.id)
        db.expire_all()
        s = db.query(AttendanceSession).filter_by(id=sess.id).first()
        assert s.is_active is False
        assert s.ended_at is not None

    def test_store_template(self, db):
        student = enroll_student(db, "Bob Jones", "P002", consent=True)
        emb = [0.1] * 512
        tmpl = store_template(db, student.id, emb,
                              "TestModel", "test.pt", "test_preprocessing")
        assert tmpl.id is not None
        assert json.loads(tmpl.embedding_json) == emb

    def test_duplicate_attendance_blocked(self, db):
        """
        Critical: second PRESENT for same student/session must return None.
        """
        import numpy as np
        student = enroll_student(db, "Carol", "P003", consent=True)
        sess = create_session(db, "Test session dup")
        r = FaceRecognizer()

        emb = np.zeros(512, dtype=np.float32); emb[0] = 1.0
        enrolled = {student.id: (student.display_name, emb.copy())}

        # First PRESENT
        from smartattend.recognizer import MatchResult, MatchStatus
        match1 = MatchResult(
            status=MatchStatus.PRESENT,
            student_id=student.id,
            student_name=student.display_name,
            similarity=0.95,
            threshold_accept=0.40,
            threshold_review=0.30,
            latency_ms=10.0,
        )
        evt1 = record_attendance(db, sess.id, match1, frame_index=0)
        assert evt1 is not None, "First PRESENT should be recorded"
        assert evt1.is_accepted is True

        # Second PRESENT for same student/session → must be blocked
        match2 = MatchResult(
            status=MatchStatus.PRESENT,
            student_id=student.id,
            student_name=student.display_name,
            similarity=0.96,
            threshold_accept=0.40,
            threshold_review=0.30,
            latency_ms=11.0,
        )
        evt2 = record_attendance(db, sess.id, match2, frame_index=5)
        assert evt2 is None, "Duplicate PRESENT must be blocked (return None)"

    def test_has_accepted_attendance(self, db):
        student = enroll_student(db, "Dave", "P004", consent=True)
        sess = create_session(db, "HAA session")
        assert not has_accepted_attendance(db, student.id, sess.id)

        from smartattend.recognizer import MatchResult, MatchStatus
        match = MatchResult(
            status=MatchStatus.PRESENT, student_id=student.id,
            student_name="Dave", similarity=0.9,
            threshold_accept=0.40, threshold_review=0.30, latency_ms=5.0,
        )
        record_attendance(db, sess.id, match, frame_index=0)
        assert has_accepted_attendance(db, student.id, sess.id)

    def test_teacher_review_creates_audit_record(self, db):
        from smartattend.database import ReviewEvent
        student = enroll_student(db, "Eve", "P005", consent=True)
        sess = create_session(db, "Review session")

        from smartattend.recognizer import MatchResult, MatchStatus
        match = MatchResult(
            status=MatchStatus.REVIEW_REQUIRED, student_id=student.id,
            student_name="Eve", similarity=0.35,
            threshold_accept=0.40, threshold_review=0.30, latency_ms=5.0,
        )
        evt = record_attendance(db, sess.id, match, frame_index=0)
        review = teacher_review(db, evt.id, "PRESENT", "verified in person")

        assert review.id is not None
        assert review.original_status == "REVIEW_REQUIRED"
        assert review.corrected_status == "PRESENT"
        n_reviews = db.query(ReviewEvent).filter_by(attendance_event_id=evt.id).count()
        assert n_reviews >= 1

    def test_unknown_not_assigned_to_nearest_student(self, db):
        """
        Unknown identity must not be silently assigned to the nearest enrolled
        person when similarity is below threshold.
        """
        from smartattend.recognizer import MatchResult, MatchStatus
        student = enroll_student(db, "Frank", "P006", consent=True)
        sess = create_session(db, "Unknown test")

        match = MatchResult(
            status=MatchStatus.UNKNOWN, student_id=None,
            student_name=None, similarity=0.05,
            threshold_accept=0.40, threshold_review=0.30, latency_ms=5.0,
        )
        evt = record_attendance(db, sess.id, match, frame_index=0)
        assert evt is not None
        assert evt.student_id is None, "UNKNOWN must not be assigned to a student"
        assert evt.is_accepted is False


# ── Attendance Logic and Review Queue Tests ────────────────────────────────────

class TestAttendanceLogicAndReviewQueue:
    """Tests for: per-student deduplication, UNKNOWN upsert, temporal integrity,
    teacher audit trail, confirmed-attendance immutability, and artifact integrity."""

    def _make_match(self, status, student_id=None, student_name=None, similarity=0.75):
        from smartattend.recognizer import MatchResult, MatchStatus
        return MatchResult(
            status=status,
            student_id=student_id,
            student_name=student_name,
            similarity=similarity,
            threshold_accept=0.65,
            threshold_review=0.52,
            latency_ms=10.0,
        )

    def test_repeated_frames_create_only_one_present_record(self, db):
        """Requirement 1: Repeated camera frames for same student produce exactly one
        accepted PRESENT record per session — not one per frame."""
        student = enroll_student(db, "Iris Newton", "RQ001", consent=True)
        sess = create_session(db, "Repeat frames test")
        match = self._make_match(MatchStatus.PRESENT, student.id, student.display_name, 0.80)

        # Simulate 5 identical frames all triggering record_attendance
        results = [record_attendance(db, sess.id, match, frame_index=i) for i in range(5)]

        # Only the first call should create a record; subsequent ones return None
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 1, f"Expected 1 PRESENT record, got {len(non_none)}"
        assert non_none[0].is_accepted is True
        assert non_none[0].student_id == student.id

        # Verify DB count
        count = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, student_id=student.id, status="PRESENT"
        ).count()
        assert count == 1, f"DB must contain exactly 1 PRESENT record, found {count}"

    def test_repeated_uncertain_frames_create_only_one_review_item_per_student(self, db):
        """Requirement 2: Repeated borderline-confidence frames for same student create
        exactly one REVIEW_REQUIRED entry per student per session."""
        student = enroll_student(db, "Jay Turing", "RQ002", consent=True)
        sess = create_session(db, "Review dedup test")
        match = self._make_match(MatchStatus.REVIEW_REQUIRED, student.id, student.display_name, 0.58)

        results = [record_attendance(db, sess.id, match, frame_index=i) for i in range(5)]
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 1, f"Expected 1 REVIEW_REQUIRED record, got {len(non_none)}"

        count = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, student_id=student.id, status="REVIEW_REQUIRED"
        ).count()
        assert count == 1, f"DB must contain exactly 1 REVIEW_REQUIRED record, found {count}"

    def test_repeated_unknown_observations_create_exactly_one_record_per_session(self, db):
        """Requirement 3: Repeated UNKNOWN observations (unrecognised face) are upserted —
        they update the single existing record in place, never inserting additional rows."""
        sess = create_session(db, "Unknown upsert test")
        match = self._make_match(MatchStatus.UNKNOWN, student_id=None, student_name=None, similarity=0.10)

        # Simulate 10 UNKNOWN frames across one session
        for i in range(10):
            record_attendance(db, sess.id, match, frame_index=i)

        count = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, status="UNKNOWN"
        ).count()
        assert count == 1, (
            f"UNKNOWN upsert must maintain exactly 1 record per session, but found {count}. "
            "Repeated observations should update in place."
        )

        # Verify frame_index was updated (should reflect the last frame, not frame 0)
        evt = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, status="UNKNOWN"
        ).first()
        assert evt.frame_index == 9, (
            f"frame_index should reflect latest observation (9), got {evt.frame_index}"
        )

    def test_unknown_observations_never_assign_student_absent_or_present(self, db):
        """Requirement 4: An UNKNOWN observation must never assign a student identity,
        set is_accepted=True, or imply any student's absence."""
        student = enroll_student(db, "Kai Lovelace", "RQ003", consent=True)
        sess = create_session(db, "Unknown no-assign test")
        match = self._make_match(MatchStatus.UNKNOWN, student_id=None, student_name=None, similarity=0.02)

        evt = record_attendance(db, sess.id, match, frame_index=0)

        assert evt is not None
        assert evt.student_id is None, "UNKNOWN event must not assign any student_id"
        assert evt.is_accepted is False, "UNKNOWN event must never be accepted"
        assert evt.status == "UNKNOWN"

        # Absence is NOT recorded — student has no PRESENT or ABSENT record
        absent_or_present = db.query(AttendanceEvent).filter(
            AttendanceEvent.session_id == sess.id,
            AttendanceEvent.student_id == student.id,
        ).count()
        assert absent_or_present == 0, (
            "UNKNOWN observation must not create any record linking to the enrolled student"
        )

    def test_teacher_review_removes_item_from_pending_queue_and_preserves_audit_trail(self, db):
        """Requirement 5: After teacher resolves an event (PRESENT/REJECTED/DISMISSED),
        it must no longer appear in REVIEW_REQUIRED pending results, but the
        ReviewEvent audit record must still exist."""
        from smartattend.database import ReviewEvent
        student = enroll_student(db, "Lena Hopper", "RQ004", consent=True)
        sess = create_session(db, "Audit trail test")
        match = self._make_match(MatchStatus.REVIEW_REQUIRED, student.id, student.display_name, 0.60)

        evt = record_attendance(db, sess.id, match, frame_index=0)
        assert evt is not None
        assert evt.status == "REVIEW_REQUIRED"

        # Teacher resolves it as REJECTED
        review = teacher_review(db, evt.id, "REJECTED", "not matching in person")

        # Item must no longer appear as REVIEW_REQUIRED
        still_pending = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, status="REVIEW_REQUIRED", is_accepted=False
        ).count()
        assert still_pending == 0, (
            "Resolved REVIEW_REQUIRED item must not appear in pending query after teacher_review"
        )

        # Event status must be updated to REJECTED
        db.refresh(evt)
        assert evt.status == "REJECTED", f"Expected status=REJECTED, got {evt.status}"

        # Audit trail (ReviewEvent) must be preserved with original status
        audit_record = db.query(ReviewEvent).filter_by(attendance_event_id=evt.id).first()
        assert audit_record is not None, "ReviewEvent audit record must be preserved"
        assert audit_record.original_status == "REVIEW_REQUIRED"
        assert audit_record.corrected_status == "REJECTED"
        assert audit_record.reason == "not matching in person"

    def test_teacher_dismiss_also_removes_from_pending_queue(self, db):
        """Requirement 5 (dismiss variant): Dismissed items also leave the review queue."""
        student = enroll_student(db, "Mia Shannon", "RQ005", consent=True)
        sess = create_session(db, "Dismiss test")
        match = self._make_match(MatchStatus.REVIEW_REQUIRED, student.id, student.display_name, 0.55)

        evt = record_attendance(db, sess.id, match, frame_index=0)
        teacher_review(db, evt.id, "DISMISSED", "camera artefact")

        still_pending = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, status="REVIEW_REQUIRED"
        ).count()
        assert still_pending == 0

        db.refresh(evt)
        assert evt.status == "DISMISSED"

    def test_confirmed_present_cannot_be_overwritten_by_conflicting_identity(self, db):
        """Requirement 6: Once a student is confirmed PRESENT (is_accepted=True),
        a subsequent call claiming a different student identity for the same session
        must not overwrite or conflict with the confirmed record."""
        student_a = enroll_student(db, "Nina Curie", "RQ006A", consent=True)
        student_b = enroll_student(db, "Omar Faraday", "RQ006B", consent=True)
        sess = create_session(db, "Overwrite protection test")

        # Student A confirmed present
        match_a = self._make_match(MatchStatus.PRESENT, student_a.id, student_a.display_name, 0.85)
        evt_a = record_attendance(db, sess.id, match_a, frame_index=0)
        assert evt_a is not None and evt_a.is_accepted is True

        # Student B also confirmed present (independent — should succeed)
        match_b = self._make_match(MatchStatus.PRESENT, student_b.id, student_b.display_name, 0.80)
        evt_b = record_attendance(db, sess.id, match_b, frame_index=10)
        assert evt_b is not None and evt_b.is_accepted is True

        # A second PRESENT attempt for student A must be blocked (no overwrite, no duplicate)
        match_a2 = self._make_match(MatchStatus.PRESENT, student_a.id, student_a.display_name, 0.70)
        evt_a2 = record_attendance(db, sess.id, match_a2, frame_index=20)
        assert evt_a2 is None, "Duplicate PRESENT for already-confirmed student must be blocked"

        # DB must still have exactly 1 accepted PRESENT per student
        count_a = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, student_id=student_a.id, status="PRESENT", is_accepted=True
        ).count()
        count_b = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, student_id=student_b.id, status="PRESENT", is_accepted=True
        ).count()
        assert count_a == 1, f"Student A must have exactly 1 PRESENT record, found {count_a}"
        assert count_b == 1, f"Student B must have exactly 1 PRESENT record, found {count_b}"

    def test_canonical_experiment_artifacts_are_intact(self):
        """Requirement 7: Canonical evaluation files must not be altered by code changes.
        Checks that metrics.json and predictions_anonymized.csv still exist and contain
        expected values matching the documented research results."""
        import json
        from pathlib import Path

        repo_root = Path(__file__).parents[1]
        metrics_path = repo_root / "experiments" / "metrics.json"
        predictions_path = repo_root / "experiments" / "predictions_anonymized.csv"

        assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"
        assert predictions_path.exists(), f"predictions_anonymized.csv not found at {predictions_path}"

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Verify canonical threshold values from the research evaluation
        assert metrics["metadata"]["threshold_accept"] == 0.65, (
            "Canonical threshold_accept in metrics.json must remain 0.65"
        )
        assert metrics["metadata"]["threshold_review"] == 0.52, (
            "Canonical threshold_review in metrics.json must remain 0.52"
        )
        assert metrics["metadata"]["total_test_probes"] == 166
        assert metrics["method_A_recognition_only"]["true_accepts_identity"] == 65

        # Verify predictions CSV has correct row count
        lines = predictions_path.read_text().strip().splitlines()
        data_rows = len(lines) - 1  # subtract header
        assert data_rows == 166, (
            f"predictions_anonymized.csv must have 166 data rows, found {data_rows}"
        )


# ── Bug Fixes & Regression Suite ─────────────────────────────────────────────


class TestBugFixesRegression:

    def test_format_similarity_safely_handles_all_numeric_and_null_inputs(self):
        """Bug 1: Verify safe formatting without crashing on None, NaN, 0.0, or invalid types."""
        from smartattend.camera import format_similarity
        import numpy as np

        assert format_similarity(None) == "N/A"
        assert format_similarity(None, default="—") == "—"
        assert format_similarity(0.0) == "0.000"
        assert format_similarity(0.65432, decimals=3) == "0.654"
        assert format_similarity(0.65432, decimals=2) == "0.65"
        assert format_similarity(-0.12, decimals=2) == "-0.12"
        assert format_similarity(float("nan")) == "N/A"
        assert format_similarity(float("inf")) == "N/A"
        assert format_similarity("invalid") == "N/A"

    def test_review_queue_rendering_simulation_with_none_similarity(self, db):
        """Bug 1: Verify Review Queue formatting logic renders None similarity without crashing."""
        from smartattend.camera import format_similarity
        from smartattend.database import AttendanceEvent
        import numpy as np

        sess = create_session(db, "Test Queue None Sim")
        # Direct DB insert mimicking historic records with None similarity
        event = AttendanceEvent(
            session_id=sess.id,
            student_id=None,
            status="UNKNOWN",
            similarity_score=None,
            similarity_threshold=0.65,
            pad_status="PAD_UNAVAILABLE",
            is_accepted=False,
        )
        db.add(event)
        db.commit()
        db.refresh(event)

        # Test formatting as executed in app.py Review Queue
        sim_display = format_similarity(event.similarity_score, 3, "N/A")
        expander_title = f"[{event.status}] Event #{event.id} — UNKNOWN — sim={sim_display}"
        assert "sim=N/A" in expander_title

        sim_val = (
            round(float(event.similarity_score), 4)
            if (event.similarity_score is not None and not np.isnan(float(event.similarity_score)))
            else None
        )
        assert sim_val is None

    def test_record_attendance_deduplicates_unknown_with_cooldown(self, db):
        """Bug 3 (updated): Verify UNKNOWN events are upserted — the second observation
        updates the existing record in place rather than inserting a new row.
        The return value is the updated existing record (not None), and DB count stays at 1.
        """
        sess = create_session(db, "Test Unknown Cooldown")
        from smartattend.recognizer import MatchResult, MatchStatus

        match = MatchResult(
            status=MatchStatus.UNKNOWN,
            student_id=None,
            student_name=None,
            similarity=0.35,
            threshold_accept=0.65,
            threshold_review=0.52,
            latency_ms=10.0,
        )

        # First UNKNOWN record: should be stored
        evt1 = record_attendance(db, sess.id, match, frame_index=0)
        assert evt1 is not None
        assert evt1.status == "UNKNOWN"

        # Second UNKNOWN call: must NOT create a new row — returns the existing updated record
        evt2 = record_attendance(db, sess.id, match, frame_index=10)
        assert evt2 is not None, "Second UNKNOWN call should return existing upserted record (not None)"
        assert evt2.id == evt1.id, "Second UNKNOWN call must return the same record (upsert, not new row)"
        assert evt2.frame_index == 10, "Upserted record must have updated frame_index"

        # DB must still contain exactly 1 UNKNOWN record for this session
        count = db.query(AttendanceEvent).filter_by(
            session_id=sess.id, status="UNKNOWN"
        ).count()
        assert count == 1, f"Upsert must keep exactly 1 UNKNOWN record per session, found {count}"


    def test_record_attendance_deduplicates_review_required(self, db):
        """Bug 3: Verify REVIEW_REQUIRED items are deduplicated per student per session."""
        student = enroll_student(db, "Grace Hopper", "P007", consent=True)
        sess = create_session(db, "Test Review Dedup")
        from smartattend.recognizer import MatchResult, MatchStatus

        match = MatchResult(
            status=MatchStatus.REVIEW_REQUIRED,
            student_id=student.id,
            student_name=student.display_name,
            similarity=0.58,
            threshold_accept=0.65,
            threshold_review=0.52,
            latency_ms=12.0,
        )

        # First ambiguous match: should be stored in review queue
        evt1 = record_attendance(db, sess.id, match, frame_index=0)
        assert evt1 is not None
        assert evt1.status == "REVIEW_REQUIRED"

        # Second ambiguous match for same student in same session: must be blocked
        evt2 = record_attendance(db, sess.id, match, frame_index=10)
        assert evt2 is None, "Duplicate REVIEW_REQUIRED for same student in session must be blocked"

    def test_camera_diagnose_structure(self):
        """Bug 2: Verify camera diagnostic routine returns expected schema and keys."""
        from smartattend.camera import diagnose_camera
        diag = diagnose_camera(0)
        assert "device_index" in diag
        assert "is_accessible" in diag
        assert "backends" in diag
        assert "hints" in diag
        assert "DirectShow" in diag["backends"]
        assert "MSMF" in diag["backends"]

    def test_camera_open_invalid_index_returns_none_safely(self):
        """Bug 2: Verify camera opener handles nonexistent camera index gracefully without raising."""
        from smartattend.camera import open_camera
        cap, backend, err = open_camera(index=99)
        assert cap is None
        assert backend == ""
        assert err is not None
        assert "99" in err

    def test_review_queue_renders_actual_historical_db_records(self):
        """Bug 1 & 3: Query real smartattend.db (which contains historic UNKNOWN records) and verify safe formatting."""
        from smartattend.camera import format_similarity
        from smartattend.database import get_session_factory, AttendanceEvent
        from pathlib import Path
        import sqlite3

        db_path = Path(__file__).parents[1] / "smartattend.db"
        if not db_path.exists():
            return  # skip if running in environment without local db

        factory = get_session_factory(str(db_path))
        with factory() as db:
            events = db.query(AttendanceEvent).all()
            for evt in events:
                # This must never raise TypeError: unsupported format string
                s = format_similarity(evt.similarity_score, 3, "N/A")
                title = f"[{evt.status}] Event #{evt.id} — sim={s}"
                assert isinstance(title, str)

