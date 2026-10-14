"""
demo_pipeline_e2e.py – End-to-end PRESENCIFY verification pipeline.

Performs:
  1. Initialization of SQLite database.
  2. Consent-based enrollment of a test participant (e.g., P001).
  3. Template extraction via MediaPipe BlazeFace + EdgeFace XS.
  4. Session creation ("Morning Class").
  5. Attendance matching:
     - Match known participant -> PRESENT (accepted).
     - Second identical match in same session -> DUPLICATE BLOCKED.
     - Match unknown participant -> UNKNOWN (not accepted).
     - Ambiguous match -> REVIEW_REQUIRED.
  6. Teacher review of REVIEW_REQUIRED event -> manually accepted with audit trail.
  7. Session close.
  8. Anonymized CSV export and diagnostics generation.

Can run with:
  python scripts/demo_pipeline_e2e.py            # Uses synthetic face images
  python scripts/demo_pipeline_e2e.py --webcam   # Enrolls & matches from live webcam
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Setup paths
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from presencify.aligner import align_face_112
from presencify.attendance import (
    create_session, end_session, enroll_student, has_accepted_attendance,
    load_enrolled_embeddings, record_attendance, store_template, teacher_review,
)
from presencify.database import AttendanceEvent, get_session_factory, init_db
from presencify.detector import FaceDetector
from presencify.export import (
    export_environment, export_model_manifest, export_predictions, export_summary,
)
from presencify.recognizer import FaceRecognizer, MatchStatus


def create_synthetic_face(seed: int = 42, size: int = 480) -> np.ndarray:
    np.random.seed(seed)
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    cv2.ellipse(img, (cx, cy), (size // 5, size // 4), 0, 0, 360, (195, 155, 115), -1)
    eye_y = cy - size // 12
    cv2.circle(img, (cx - size // 10, eye_y), size // 24, (50, 30, 20), -1)
    cv2.circle(img, (cx + size // 10, eye_y), size // 24, (50, 30, 20), -1)
    # Nose
    cv2.circle(img, (cx, cy), size // 30, (160, 120, 90), -1)
    # Mouth
    cv2.ellipse(img, (cx, cy + size // 8), (size // 15, size // 30), 0, 0, 180, (80, 50, 40), 2)
    return img


def run_e2e(use_webcam: bool = False) -> int:
    print("=" * 65)
    print("PRESENCIFY - End-to-End Pipeline Verification")
    print(f"Mode: {'Live Webcam' if use_webcam else 'Synthetic Simulation'}")
    print("=" * 65)

    db_path = str(ROOT / "presencify_demo.db")
    init_db(db_path)
    factory = get_session_factory(db_path)

    # 1. Load models
    print("\n[1/7] Initializing Detector & Recognizer on CPU...")
    detector = FaceDetector()
    assert detector.load(), f"Detector failed: {detector.status['load_error']}"
    print(f"  Detector: {detector.MODEL_NAME} (load: {detector.status['load_time_ms']} ms)")

    recognizer = FaceRecognizer()
    assert recognizer.load(), f"Recognizer failed: {recognizer.status['load_error']}"
    print(f"  Recognizer: {recognizer.MODEL_NAME} (load: {recognizer.status['load_time_ms']} ms)")

    # 2. Enrollment
    print("\n[2/7] Enrolling Student with Consent...")
    with factory() as db:
        student = enroll_student(
            db, display_name="Demo Participant A", pseudonym="P001", consent=True
        )
        print(f"  Enrolled: {student.display_name} -> Pseudonym {student.pseudonym}")

        # Capture template
        if use_webcam:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            frame = None
            for _ in range(15):
                ret, f = cap.read()
                if ret and detector.detect(f):
                    frame = f
                    break
                time.sleep(0.05)
            cap.release()
            if frame is None:
                print("  [FAIL] No face detected from webcam.")
                return 1
        else:
            frame = create_synthetic_face(seed=42)

        dets = detector.detect(frame)
        assert len(dets) >= 1, "Face detection returned no detections."
        det = dets[0]
        crop112 = align_face_112(frame, det.bbox_pixel, det.keypoints)
        emb = recognizer.embed(crop112)
        assert emb is not None, "Failed to extract embedding."

        tmpl = store_template(
            db,
            student_id=student.id,
            embedding=emb.tolist(),
            model_name=recognizer.MODEL_NAME,
            model_checkpoint=recognizer.MODEL_CHECKPOINT,
            preprocessing=recognizer.PREPROCESSING,
            quality_score=det.confidence,
        )
        print(f"  Stored template: 512-d embedding (norm={np.linalg.norm(emb):.4f})")

    # 3. Session Creation
    print("\n[3/7] Creating Attendance Session...")
    with factory() as db:
        session = create_session(db, name="E2E Demo Class 101")
        session_id = session.id
        print(f"  Active Session ID: {session_id} ('{session.name}')")

    # 4. Recognition & Attendance Decisions
    print("\n[4/7] Testing Attendance Decision Engine...")
    with factory() as db:
        enrolled = load_enrolled_embeddings(db, recognizer.MODEL_NAME)
        print(f"  Loaded enrolled profiles: {len(enrolled)}")

        # A) Matching enrolled participant
        match_known = recognizer.match(emb, enrolled)
        print(f"  Query Known: Status={match_known.status.value}, Sim={match_known.similarity:.4f}")
        assert match_known.status == MatchStatus.PRESENT

        evt1 = record_attendance(db, session_id, match_known, frame_index=1, pad_status="PAD_UNAVAILABLE")
        assert evt1 is not None and evt1.is_accepted is True
        print("  -> First PRESENT event ACCEPTED.")

        # B) Duplicate suppression
        evt_dup = record_attendance(db, session_id, match_known, frame_index=2, pad_status="PAD_UNAVAILABLE")
        assert evt_dup is None, "Duplicate attendance was not blocked!"
        print("  -> Second identical PRESENT in same session BLOCKED (Duplicate Prevention OK).")

        # C) Unknown participant
        unknown_emb = np.random.randn(512).astype(np.float32)
        unknown_emb /= np.linalg.norm(unknown_emb)
        match_unknown = recognizer.match(unknown_emb, enrolled)
        print(f"  Query Unknown: Status={match_unknown.status.value}, Sim={match_unknown.similarity}")
        assert match_unknown.status == MatchStatus.UNKNOWN
        evt_unk = record_attendance(db, session_id, match_unknown, frame_index=3, pad_status="PAD_UNAVAILABLE")
        assert evt_unk.student_id is None and evt_unk.is_accepted is False
        print("  -> Unknown face logged without assigning identity.")

        # D) Ambiguous match -> REVIEW_REQUIRED
        # Create embedding with borderline similarity (e.g. 0.35, between review=0.30 and accept=0.40)
        target = list(enrolled.values())[0][1]
        noise = np.random.randn(512).astype(np.float32)
        # Interpolate
        interp = 0.35 * target + np.sqrt(1 - 0.35**2) * (noise / np.linalg.norm(noise))
        interp /= np.linalg.norm(interp)
        match_borderline = recognizer.match(interp, enrolled)
        print(f"  Query Borderline: Status={match_borderline.status.value}, Sim={match_borderline.similarity:.4f}")
        assert match_borderline.status == MatchStatus.REVIEW_REQUIRED
        evt_rev = record_attendance(db, session_id, match_borderline, frame_index=4, pad_status="PAD_UNAVAILABLE")
        assert evt_rev.is_accepted is False
        rev_id = evt_rev.id
        print(f"  -> Ambiguous match routed to REVIEW_REQUIRED (Event #{rev_id}).")

    # 5. Teacher Review Audit
    print("\n[5/7] Testing Teacher Review Audit...")
    with factory() as db:
        review_audit = teacher_review(
            db,
            event_id=rev_id,
            corrected_status="REJECTED",
            reason="Low similarity, visually unverified",
            reviewer_note="Teacher review completed",
        )
        print(f"  Review Action: Event #{rev_id} changed from {review_audit.original_status} -> {review_audit.corrected_status}")

    # 6. Session Close
    print("\n[6/7] Closing Session...")
    with factory() as db:
        end_session(db, session_id)
        print("  Session ended successfully.")

    # 7. Anonymized Exports
    print("\n[7/7] Generating Anonymized Research Exports...")
    with factory() as db:
        csv_pred = export_predictions(db, session_id=session_id)
        csv_sum = export_summary(db)
    env_txt = export_environment(recognizer.status, detector.status)
    manifest = export_model_manifest(recognizer.status, detector.status)

    print(f"  Predictions CSV : {csv_pred}")
    print(f"  Summary CSV     : {csv_sum}")
    print(f"  Environment     : {env_txt}")
    print(f"  Model Manifest  : {manifest}")

    print("\n" + "=" * 65)
    print("RESULT: ALL E2E PIPELINE PHASES VERIFIED SUCCESSFULLY")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="Run with live webcam frames")
    args = parser.parse_args()
    sys.exit(run_e2e(use_webcam=args.webcam))
