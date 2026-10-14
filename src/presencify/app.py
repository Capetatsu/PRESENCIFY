"""
app.py – PRESENCIFY Streamlit Dashboard (Phase 7).

Pages:
  1. Live Attendance – webcam feed with detection/recognition overlay
  2. Enroll Student  – consent + name + multiple captures
  3. Review Queue    – teacher review of uncertain/flagged events
  4. Session History – past sessions and attendance records
  5. Export          – CSV export + environment dump
  6. Diagnostics     – model status, versions, latency

Run:
  python -m streamlit run src/presencify/app.py
  (or: streamlit run src/presencify/app.py)

Non-negotiable rules enforced here:
  - Similarity shown as "similarity score" never "probability"
  - PAD status always shown; PAD_UNAVAILABLE if no model is running
  - No fake/synthetic predictions in live view
  - Duplicate attendance blocked in UI + DB
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from presencify.attendance import (
    create_session, end_session, enroll_student, load_enrolled_embeddings,
    record_attendance, store_template, teacher_review,
)
from presencify.aligner import align_face_112
from presencify.camera import (
    diagnose_camera, format_similarity, open_camera, read_frame_with_retry, release_camera,
)
from presencify.database import (
    AttendanceEvent, AttendanceSession, ReviewEvent, Student,
    get_session_factory, init_db,
)
from presencify.detector import FaceDetector
from presencify.export import (
    export_environment, export_model_manifest, export_predictions, export_summary,
)
from presencify.recognizer import FaceRecognizer, MatchStatus

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRESENCIFY",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pure Black Dark Theme Styling ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global Application Canvas: Neutral Pure Black */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #000000 !important;
        color: #F5F5F5 !important;
    }

    [data-testid="stHeader"] {
        background-color: #000000 !important;
        border-bottom: 1px solid #292929 !important;
    }

    /* Sidebar Canvas */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #080808 !important;
        border-right: 1px solid #292929 !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #F5F5F5 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    p, span, label, div {
        color: #F5F5F5;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #A3A3A3 !important;
        font-size: 0.85rem !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid #292929 !important;
        margin: 1.25rem 0 !important;
    }

    /* Form Controls: Inputs and Selects */
    .stTextInput input, .stNumberInput input, textarea {
        background-color: #161616 !important;
        color: #F5F5F5 !important;
        border: 1px solid #292929 !important;
        border-radius: 4px !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus, textarea:focus {
        border-color: #E53935 !important;
        box-shadow: 0 0 0 1px #E53935 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #161616 !important;
        color: #F5F5F5 !important;
        border: 1px solid #292929 !important;
        border-radius: 4px !important;
    }

    [data-baseweb="popover"], [data-baseweb="menu"] {
        background-color: #161616 !important;
        border: 1px solid #292929 !important;
    }

    [data-baseweb="menu"] * {
        color: #F5F5F5 !important;
    }

    /* Buttons: Secondary and Primary */
    .stButton > button {
        background-color: #161616 !important;
        color: #F5F5F5 !important;
        border: 1px solid #292929 !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.9rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }

    .stButton > button:hover {
        background-color: #242424 !important;
        border-color: #E53935 !important;
        color: #FFFFFF !important;
    }

    .stButton > button:active {
        background-color: #E53935 !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #E53935 !important;
        border: 1px solid #E53935 !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #C62828 !important;
        border-color: #C62828 !important;
    }

    /* Cards, Forms, and Expanders */
    [data-testid="stExpander"] {
        background-color: #101010 !important;
        border: 1px solid #292929 !important;
        border-radius: 4px !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stExpander"] summary {
        color: #F5F5F5 !important;
        font-weight: 500 !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: #E53935 !important;
    }

    [data-testid="stForm"] {
        background-color: #101010 !important;
        border: 1px solid #292929 !important;
        border-radius: 6px !important;
        padding: 1.25rem !important;
    }

    /* Alerts and Status Banners */
    [data-testid="stAlert"] {
        background-color: #101010 !important;
        border: 1px solid #292929 !important;
        border-radius: 4px !important;
        color: #F5F5F5 !important;
    }

    [data-testid="stAlert"] div {
        color: #F5F5F5 !important;
    }

    /* JSON, Code, and Preformatted blocks */
    [data-testid="stJson"], pre, code {
        background-color: #101010 !important;
        border: 1px solid #292929 !important;
        border-radius: 4px !important;
        color: #F5F5F5 !important;
    }

    /* Sidebar Radio Navigation */
    [data-testid="stRadio"] label {
        color: #F5F5F5 !important;
        cursor: pointer !important;
    }

    [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.35rem !important;
    }

    /* DataTables and DataFrames */
    [data-testid="stDataFrame"] {
        background-color: #101010 !important;
        border: 1px solid #292929 !important;
        border-radius: 4px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH = str(Path(__file__).parents[2] / "presencify.db")

@st.cache_resource
def get_db():
    init_db(DB_PATH)
    return get_session_factory(DB_PATH)

@st.cache_resource
def get_detector_instance():
    d = FaceDetector()
    d.load()
    return d

@st.cache_resource
def get_recognizer_instance():
    r = FaceRecognizer()
    r.load()
    return r

SessionFactory = get_db()
detector = get_detector_instance()
recognizer = get_recognizer_instance()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("PRESENCIFY")
st.sidebar.caption("Research prototype | CPU-only | No CUDA")

page = st.sidebar.radio(
    "Navigation",
    ["Live Attendance", "Enroll Student",
     "Review Queue", "Session History",
     "Export", "Diagnostics"],
)

# PAD status banner (always visible)
st.sidebar.markdown("---")
st.sidebar.warning(
    "**PAD: UNAVAILABLE**\n\n"
    "No presentation-attack detection model is loaded. "
    "The system cannot detect spoofing (printed photos, replays). "
    "All decisions show PAD_UNAVAILABLE."
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Attendance
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Live Attendance":
    st.title("Live Attendance")

    with get_db()() as db:
        sessions = db.query(AttendanceSession).filter_by(is_active=True).all()
        session_opts = {s.name: s.id for s in sessions}

    col_ctrl, col_frame = st.columns([1, 2])

    with col_ctrl:
        if session_opts:
            chosen_name = st.selectbox("Active session", list(session_opts.keys()))
            active_session_id = session_opts[chosen_name]
        else:
            st.info("No active session. Create one below.")
            active_session_id = None

        new_name = st.text_input("New session name")
        if st.button("Start session") and new_name.strip():
            with get_db()() as db:
                s = create_session(db, new_name.strip())
                active_session_id = s.id
            st.success(f"Session '{new_name}' started.")
            st.rerun()

        if active_session_id:
            if st.button("End current session"):
                with get_db()() as db:
                    end_session(db, active_session_id)
                st.success("Session ended.")
                st.rerun()

        st.markdown("---")
        with st.expander("Camera Settings and Diagnostics", expanded=False):
            cam_idx = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
            cam_backend_opt = st.selectbox(
                "Preferred backend",
                ["Auto (DirectShow -> MSMF -> ANY)", "DirectShow", "MSMF", "ANY"],
            )
            if st.button("Run Camera Diagnostic", key="btn_diag_cam"):
                diag = diagnose_camera(int(cam_idx))
                if diag["is_accessible"]:
                    st.success(f"Camera index {cam_idx} accessible using '{diag['recommended_backend']}'.")
                else:
                    st.error(f"Camera index {cam_idx} could not be opened on any backend.")
                st.json(diag)
                for h in diag.get("hints", []):
                    st.caption(f"- {h}")

        run_live = st.toggle("Run webcam", value=False)
        frame_delay = st.slider("Frame skip (ms)", 50, 500, 100)

    with col_frame:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

    if run_live and active_session_id:
        with get_db()() as db:
            enrolled = load_enrolled_embeddings(db, recognizer.MODEL_NAME)

        has_enrolled = len(enrolled) > 0
        if not has_enrolled:
            st.warning(
                "No students enrolled yet. Face recognition requires at least one enrolled student. "
                "Go to 'Enroll Student' to add participants before taking attendance."
            )

        backend_pref = None if "Auto" in cam_backend_opt else (
            "DirectShow" if "DirectShow" in cam_backend_opt else ("MSMF" if "MSMF" in cam_backend_opt else "ANY")
        )
        cap, actual_backend, open_err = open_camera(index=int(cam_idx), preferred_backend=backend_pref)

        if cap is None:
            st.error(f"Camera initialization failed on device index {cam_idx}: {open_err}")
            diag = diagnose_camera(int(cam_idx))
            with st.expander("Camera Diagnostic Details and Troubleshooting", expanded=True):
                st.json(diag)
                for h in diag.get("hints", []):
                    st.info(f"{h}")
        else:
            frame_idx = 0
            consecutive_read_failures = 0
            MAX_CONSECUTIVE_FAILURES = 5

            # Track already-accepted students to avoid re-querying DB every frame
            already_present: set[str] = set()
            with get_db()() as db:
                accepted = db.query(AttendanceEvent).filter_by(
                    session_id=active_session_id, is_accepted=True, status="PRESENT"
                ).all()
                already_present = {e.student_id for e in accepted if e.student_id}

            # Temporal consistency state: require N_CONSISTENT consecutive frames
            # with the same student identity before persisting a PRESENT event.
            # This prevents a single transient frame from triggering attendance.
            N_CONSISTENT = 3   # configurable; do not tune against the test set
            _consec_student_id: Optional[str] = None
            _consec_count: int = 0

            try:
                while run_live:
                    ret, frame = read_frame_with_retry(cap, max_retries=3, delay_sec=0.03)
                    if not ret or frame is None:
                        consecutive_read_failures += 1
                        if consecutive_read_failures >= MAX_CONSECUTIVE_FAILURES:
                            st.error(
                                f"Camera read failed after {MAX_CONSECUTIVE_FAILURES} consecutive attempts "
                                f"on backend '{actual_backend}'. The camera stream may have disconnected or "
                                "been claimed by another process."
                            )
                            break
                        time.sleep(0.04)
                        continue

                    consecutive_read_failures = 0  # reset on successful read
                    display = frame.copy()
                    detections = detector.detect(frame)
                    statuses = []

                    for det in detections:
                        x, y, w, h = det.bbox_pixel
                        face_112 = align_face_112(frame, det.bbox_pixel, det.keypoints)
                        embedding = recognizer.embed(face_112)

                        if embedding is None:
                            label = "MODEL_ERROR"
                            color = (128, 128, 128)
                        elif not has_enrolled:
                            label = "UNENROLLED_SYSTEM | Enroll students first | PAD:UNAVAIL"
                            color = (0, 165, 255)
                            statuses.append({
                                "status": "UNENROLLED_SYSTEM",
                                "message": "No students enrolled in database. Enroll students first.",
                            })
                        else:
                            match = recognizer.match(embedding, enrolled)

                            # ── Temporal consistency guard ────────────────────────
                            # Only persist a PRESENT decision after N_CONSISTENT
                            # consecutive frames agree on the same student identity.
                            # Protects against single-frame transient matches.
                            # REVIEW_REQUIRED and UNKNOWN are persisted per-frame gate
                            # (deduplicated in record_attendance).
                            if match.status == MatchStatus.PRESENT and match.student_id:
                                if match.student_id not in already_present:
                                    # Accumulate consistent-identity evidence
                                    if match.student_id == _consec_student_id:
                                        _consec_count += 1
                                    else:
                                        _consec_student_id = match.student_id
                                        _consec_count = 1

                                    if _consec_count >= N_CONSISTENT:
                                        # Sufficient evidence — persist attendance
                                        with get_db()() as db:
                                            evt = record_attendance(
                                                db, active_session_id, match,
                                                frame_index=frame_idx,
                                                pad_status="PAD_UNAVAILABLE",
                                            )
                                        if evt is not None:
                                            already_present.add(match.student_id)
                                        # Reset consistency counter after decision
                                        _consec_student_id = None
                                        _consec_count = 0
                                    # else: still accumulating — do not persist yet
                                # else: already confirmed in this session — skip
                            else:
                                # Reset consistency counter if identity changes or non-PRESENT
                                if match.status != MatchStatus.PRESENT:
                                    _consec_student_id = None
                                    _consec_count = 0

                                # Persist REVIEW_REQUIRED and UNKNOWN (deduplicated internally)
                                if frame_idx % 10 == 0:
                                    with get_db()() as db:
                                        record_attendance(
                                            db, active_session_id, match,
                                            frame_index=frame_idx,
                                            pad_status="PAD_UNAVAILABLE",
                                        )

                            # Build overlay label
                            sim_str = format_similarity(match.similarity, 2, "N/A")
                            label = f"{match.status.value} | sim={sim_str} | PAD:UNAVAIL"
                            if match.student_name:
                                label = f"{match.student_name} | {label}"

                            # Overlay note when still accumulating consistent frames
                            if (match.status == MatchStatus.PRESENT and
                                    match.student_id not in already_present and
                                    _consec_count > 0 and _consec_count < N_CONSISTENT):
                                label += f" | confirming {_consec_count}/{N_CONSISTENT}"

                            color = {
                                MatchStatus.PRESENT:         (0, 200, 0),
                                MatchStatus.REVIEW_REQUIRED: (0, 165, 255),
                                MatchStatus.UNKNOWN:         (0, 0, 200),
                                MatchStatus.MODEL_ERROR:     (128, 128, 128),
                            }.get(match.status, (200, 200, 0))

                            statuses.append(match.to_dict())

                        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(display, label, (x, max(y-6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                    if not detections:
                        cv2.putText(display, "NO_FACE", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

                    # Add frame counter & backend status
                    cv2.putText(display, f"frame={frame_idx} | backend={actual_backend}", (10, display.shape[0]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

                    frame_placeholder.image(
                        cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_container_width=True,
                    )
                    if statuses:
                        status_placeholder.json(statuses[0])

                    frame_idx += 1
                    time.sleep(frame_delay / 1000)

            finally:
                release_camera(cap)


    elif not run_live:
        frame_placeholder.info("Toggle 'Run webcam' to start.")
    elif not active_session_id:
        frame_placeholder.warning("Start a session first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Enroll Student
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Enroll Student":
    st.title("Enroll Student")
    st.info(
        "Enrollment requires explicit consent. "
        "Only enroll people who have agreed to participate in this research prototype."
    )

    with st.form("enroll_form"):
        display_name = st.text_input("Full name (stored locally only)")
        pseudonym    = st.text_input("Pseudonym / ID (e.g. P001, used in exports)")
        consent      = st.checkbox("I confirm this participant has given informed consent to enroll")
        submitted    = st.form_submit_button("Enroll")

    if submitted:
        if not display_name.strip() or not pseudonym.strip():
            st.error("Name and pseudonym are required.")
        elif not consent:
            st.error("Consent must be confirmed.")
        else:
            with get_db()() as db:
                existing = db.query(Student).filter_by(pseudonym=pseudonym.strip()).first()
                if existing:
                    st.warning(f"Pseudonym '{pseudonym}' already enrolled. Use a different ID.")
                else:
                    student = enroll_student(db, display_name.strip(),
                                             pseudonym.strip(), consent)
                    st.session_state["enroll_student_id"] = student.id
                    st.success(f"Enrolled: {display_name} ({pseudonym}). Now capture face templates.")

    # Capture face templates for enrolled student
    if "enroll_student_id" in st.session_state:
        sid = st.session_state["enroll_student_id"]
        with get_db()() as db:
            student = db.query(Student).filter_by(id=sid).first()
            tmpl_count = db.query(Student).filter_by(id=sid).first().templates.__len__() if student else 0

        st.markdown(f"**Capturing templates for: {student.display_name if student else sid}**")
        st.caption("Capture 3–5 images from slightly different angles for best results.")

        if st.button("Capture face template"):
            cap, actual_b, open_err = open_camera(index=0, warmup_frames=3)
            if cap is None:
                st.error(f"Could not open camera device 0: {open_err}")
                diag = diagnose_camera(0)
                st.json(diag)
            else:
                captured_ok = False
                try:
                    for attempt in range(15):
                        ret, frame = read_frame_with_retry(cap, max_retries=2)
                        if not ret or frame is None:
                            continue
                        dets = detector.detect(frame)
                        if dets:
                            det = max(dets, key=lambda d: d.confidence)
                            face_112 = align_face_112(frame, det.bbox_pixel, det.keypoints)
                            embedding = recognizer.embed(face_112)
                            if embedding is not None:
                                with get_db()() as db:
                                    store_template(
                                        db, sid, embedding.tolist(),
                                        model_name=recognizer.MODEL_NAME,
                                        model_checkpoint=recognizer.MODEL_CHECKPOINT,
                                        preprocessing=recognizer.PREPROCESSING,
                                        quality_score=det.confidence,
                                    )
                                conf_str = f"{det.confidence:.2f}" if det.confidence is not None else "N/A"
                                st.image(cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB),
                                         caption=f"Captured (confidence={conf_str})",
                                         width=150)
                                st.success("Template stored.")
                                captured_ok = True
                                break
                        time.sleep(0.08)
                finally:
                    release_camera(cap)

                if not captured_ok:
                    st.error("Could not detect a clear face. Ensure your face is centered with adequate lighting.")

        if st.button("Done enrolling"):
            del st.session_state["enroll_student_id"]
            st.rerun()

    # Show enrolled students
    st.markdown("---")
    st.subheader("Currently enrolled")
    with get_db()() as db:
        students = db.query(Student).filter_by(active=True).all()
        if students:
            for s in students:
                n_tmpl = len(s.templates)
                st.write(f"- **{s.pseudonym}** ({s.display_name}) - {n_tmpl} template(s)")
        else:
            st.info("No students enrolled yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Review Queue
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Review Queue":
    st.title("Teacher Review Queue")
    st.caption(
        "Review borderline-confidence matches flagged by the model. "
        "All teacher decisions are recorded in the audit log. "
        "Camera observations of unknown faces are NOT shown here — "
        "they are raw observations and do not imply any student is absent or present."
    )

    # Session selector — show only items from the chosen session
    with get_db()() as db:
        all_sessions = db.query(AttendanceSession).order_by(
            AttendanceSession.started_at.desc()
        ).all()
        session_map = {f"{s.name} ({'Active' if s.is_active else 'Ended'})": s.id for s in all_sessions}

    if not session_map:
        st.info("No sessions found. Start a session in Live Attendance first.")
    else:
        chosen_session_label = st.selectbox(
            "Session to review", list(session_map.keys()), index=0
        )
        review_session_id = session_map[chosen_session_label]

        with get_db()() as db:
            # Only REVIEW_REQUIRED — resolved items (PRESENT/REJECTED/DISMISSED)
            # have had their status updated by teacher_review() and are excluded.
            pending = db.query(AttendanceEvent).filter(
                AttendanceEvent.status == "REVIEW_REQUIRED",
                AttendanceEvent.session_id == review_session_id,
                AttendanceEvent.is_accepted == False,
            ).order_by(AttendanceEvent.recorded_at.desc()).all()

            students = {s.id: s for s in db.query(Student).all()}

        if not pending:
            st.success("No pending review items for this session.")
        else:
            st.info(f"{len(pending)} item(s) pending teacher review.")
            for evt in pending:
                student = students.get(evt.student_id) if evt.student_id else None
                sim_display = format_similarity(evt.similarity_score, 3, "N/A")
                label_name = student.pseudonym if student else "UNKNOWN_PERSON"
                with st.expander(
                    f"[{evt.status}] Event #{evt.id} | "
                    f"{label_name} | "
                    f"sim={sim_display}"
                ):
                    sim_val = (
                        round(float(evt.similarity_score), 4)
                        if (evt.similarity_score is not None and not np.isnan(float(evt.similarity_score)))
                        else None
                    )
                    thresh_val = (
                        round(float(evt.similarity_threshold), 4)
                        if (evt.similarity_threshold is not None and not np.isnan(float(evt.similarity_threshold)))
                        else None
                    )
                    st.json({
                        "event_id": evt.id,
                        "status": evt.status,
                        "candidate_student": student.pseudonym if student else None,
                        "similarity_score": sim_val,
                        "similarity_threshold": thresh_val,
                        "review_range": f"[{thresh_val}, {thresh_val and round(thresh_val + 0.13, 2)})" if thresh_val else "N/A",
                        "pad_status": evt.pad_status,
                        "recorded_at": str(evt.recorded_at),
                        "note": (
                            "This face matched candidate student with borderline confidence. "
                            "Confirm or reject based on in-person verification."
                            if student else
                            "No candidate student matched. "
                            "Camera observation alone does not imply any student is absent."
                        ),
                    })

                    # Prevent PRESENT from being assigned to an UNKNOWN_PERSON case
                    can_mark_present = (student is not None)
                    if not can_mark_present:
                        st.warning(
                            "No enrolled student identity was linked to this event. "
                            "PRESENT cannot be assigned — dismiss or reject."
                        )

                    col1, col2, col3 = st.columns(3)
                    reason = st.text_input("Reason (optional)", key=f"reason_{evt.id}")
                    with col1:
                        if can_mark_present:
                            if st.button("Mark PRESENT", key=f"pres_{evt.id}"):
                                with get_db()() as db:
                                    teacher_review(db, evt.id, "PRESENT", reason or "verified by teacher")
                                st.success(f"Marked PRESENT for {student.pseudonym}.")
                                st.rerun()
                    with col2:
                        if st.button("Mark REJECTED", key=f"rej_{evt.id}"):
                            with get_db()() as db:
                                teacher_review(db, evt.id, "REJECTED", reason or "rejected by teacher")
                            st.success("Marked REJECTED.")
                            st.rerun()
                    with col3:
                        if st.button("Dismiss", key=f"dis_{evt.id}"):
                            with get_db()() as db:
                                teacher_review(db, evt.id, "DISMISSED", reason or "dismissed by teacher")
                            st.rerun()



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Session History
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Session History":
    st.title("Session History")

    with get_db()() as db:
        sessions = db.query(AttendanceSession).order_by(
            AttendanceSession.started_at.desc()
        ).all()

    for sess in sessions:
        with get_db()() as db:
            events = db.query(AttendanceEvent).filter_by(session_id=sess.id).all()
        present = sum(1 for e in events if e.status == "PRESENT" and e.is_accepted)
        review  = sum(1 for e in events if e.status == "REVIEW_REQUIRED")
        status_tag = "Active" if sess.is_active else "Ended"

        with st.expander(f"[{status_tag}] {sess.name} | {present} present | {review} review"):
            import pandas as pd
            rows = []
            with get_db()() as db:
                for evt in events:
                    student = db.query(Student).filter_by(id=evt.student_id).first() if evt.student_id else None
                    sim_val = (
                        round(float(evt.similarity_score), 4)
                        if (evt.similarity_score is not None and not np.isnan(float(evt.similarity_score)))
                        else None
                    )
                    rows.append({
                        "event_id": evt.id,
                        "pseudonym": student.pseudonym if student else "UNKNOWN",
                        "status": evt.status,
                        "similarity_score": sim_val,
                        "pad_status": evt.pad_status,
                        "accepted": evt.is_accepted,
                        "recorded_at": str(evt.recorded_at)[:19],
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Export
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Export":
    st.title("Export")
    st.info("All exports use pseudonyms - display names and raw frames are not included.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export predictions CSV"):
            with get_db()() as db:
                path = export_predictions(db)
            st.success(f"Saved: {path}")

        if st.button("Export attendance summary CSV"):
            with get_db()() as db:
                path = export_summary(db)
            st.success(f"Saved: {path}")

    with col2:
        if st.button("Export environment and model manifest"):
            path1 = export_environment(recognizer.status, detector.status)
            path2 = export_model_manifest(recognizer.status, detector.status)
            st.success(f"Saved:\n  {path1}\n  {path2}")

    # Show existing exports
    from presencify.export import EXPERIMENTS_DIR
    exports = list(EXPERIMENTS_DIR.glob("*")) if EXPERIMENTS_DIR.exists() else []
    if exports:
        st.markdown("**Existing exports:**")
        for f in sorted(exports):
            st.write(f"- `{f.name}` ({f.stat().st_size} bytes)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Diagnostics":
    st.title("Diagnostics")
    st.caption("Exact model and checkpoint details, device specifications, library versions, and load status.")

    import torch, cv2 as _cv2, platform

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detector")
        st.json(detector.status)
        st.subheader("Recognizer")
        st.json(recognizer.status)

    with col2:
        st.subheader("PAD")
        st.json({
            "status": "NOT_IMPLEMENTED",
            "reason": "KBS 2025 (spatio-temporal PAD) has no public code. "
                      "CASO-PAD (Sci. Reports 2026) weights unverified.",
            "ui_label": "PAD_UNAVAILABLE",
        })

        st.subheader("Camera Subsystem")
        st.json(diagnose_camera(0))

        st.subheader("Environment")
        st.json({
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.cuda.is_available(),
            "torch_cpu_threads": torch.get_num_threads(),
            "opencv": _cv2.__version__,
        })

    st.markdown("---")
    st.caption(
        "PRESENCIFY research prototype | CPU-only | "
        "EdgeFace XS (CC BY-NC-SA 4.0, non-commercial research) | "
        "MediaPipe BlazeFace (Apache 2.0)"
    )

