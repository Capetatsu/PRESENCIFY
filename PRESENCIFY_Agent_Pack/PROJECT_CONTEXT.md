# PRESENCIFY --- Project Context and System Specification

## Goal

A local research prototype for AI-assisted classroom attendance using a
webcam: detect faces, identify consenting enrolled participants, use PAD
when a verified model is available, apply confidence-based decisions,
record attendance once per session, and provide a teacher dashboard.

This is not a production campus CCTV deployment. Claims are limited to
tested participants, hardware, environment and attack types.

## Constraints

-   Windows; Intel i3-class CPU; 8 GB RAM; Intel UHD; no NVIDIA/CUDA.
-   Laptop webcam.
-   Deadline before 15 October 2026.
-   Avoid training large models, cloud dependencies and unnecessary
    services.
-   Python 3.11 64-bit is the starting environment.
-   Prefer a small local app and SQLite; use the current repository's
    stack if it already exists and works.

## Pipeline

Camera → face detection → alignment/preprocessing → embedding and
identity matching → PAD decision → confidence policy → session/duplicate
rules → database → dashboard/CSV → experiment logging.

## Required features

1.  Student enrollment with unique ID and consent confirmation.
2.  Webcam view with face boxes and reason-coded states.
3.  Named attendance sessions, start/end controls.
4.  Recognition with similarity score; do not label similarity as
    probability.
5.  PAD state and model status, if a real PAD model is integrated.
6.  States: `PRESENT`, `REVIEW_REQUIRED`, `UNKNOWN`, `REJECTED`,
    `NO_FACE`, `PAD_UNAVAILABLE`.
7.  One accepted attendance per student/session, enforced in database
    and application logic.
8.  Teacher review and correction with audit events.
9.  Attendance history and CSV export.
10. Diagnostics: exact model/checkpoint, CPU device, versions, load
    status and latency.
11. An experiment mode exporting pseudonymous predictions, conditions
    and timings.

## Stack

Python 3.11, OpenCV, PyTorch CPU where needed, SQLite/SQLAlchemy,
Streamlit for fastest local dashboard, NumPy/pandas, pytest. Use FastAPI
only if an existing architecture or real need justifies it. Do not
replace an existing stack without inspection.

## Model selection policy

### Recognition

-   Preferred research candidate: FaceLiVT, published at IEEE ICIP 2025.
    Official repository: https://github.com/novendrastywn/FaceLiVT
-   Paper DOI: https://doi.org/10.1109/ICIP55913.2025.11084611
-   Verify checkpoint availability, license, alignment, normalization,
    CPU inference and latency. Repository setup includes
    CUDA/DALI-oriented instructions; CPU loading in an example is not
    proof the entire pipeline works on CPU.
-   Comparison/fallback: EdgeFace (2024), official repo
    https://github.com/otroshi/edgeface, DOI
    https://doi.org/10.1109/TBIOM.2024.3352164. Verify exact variant,
    checkpoint, preprocessing, license and CPU route.
-   Do not headline ArcFace or FaceNet as the new model. An older loss
    or foundation may be referenced by a recent paper; distinguish the
    architecture, publication date and training objective.

### Face detection

-   Candidate: YOLO-FaceV2 (Pattern Recognition, 2024), DOI
    https://doi.org/10.1016/j.patcog.2024.110714
-   Verify runnable code, checkpoint, license and CPU speed. If
    infeasible, use a small working detector as an engineering fallback
    and disclose it.

### PAD

-   2025 candidate: "Spatio-temporal deep learning for improved face
    presentation attack detection," Knowledge-Based Systems 311 (2025),
    DOI https://doi.org/10.1016/j.knosys.2025.113059
-   2026 candidate: "Face presentation attack detection via
    content-adaptive spatial operators," Scientific Reports (2026), DOI
    https://doi.org/10.1038/s41598-026-67944-6
-   Inspect official code, supplementary files, checkpoint and
    preprocessing. Published benchmark metrics are not PRESENCIFY
    results.
-   A blink/head-turn prompt is not equivalent to a validated PAD model.
    If no recent PAD model can be run, label the fallback and limitation
    clearly.

## Decision policy

-   Calibrate thresholds on a validation split, not the final test set.
-   Require stable observations over several frames; frames must not
    create duplicate records.
-   Uncertain identity → `REVIEW_REQUIRED`.
-   PAD fail → `REJECTED`.
-   PAD unavailable → `PAD_UNAVAILABLE`; do not imply spoof protection.
-   Unknown identity → `UNKNOWN`.
-   `PRESENT` only when session is active, recognition policy passes,
    required PAD passes and no accepted record exists for that
    student/session.
-   Store original model decisions separately from teacher corrections.

## Minimal database

`students`, `face_templates`, `sessions`, `attendance_events`,
`review_events`. Store model/preprocessing versions with templates.
Enforce uniqueness for accepted `(student_id, session_id)` attendance.
Keep local-only by default; avoid retaining raw images/video unless
necessary and consented.

## Privacy

Use consenting participants. Do not scrape faces or enroll people
without consent. Keep data local; use pseudonymous IDs in experiment
exports; support deletion; do not log raw frames by default. Human
review is required for uncertain decisions.
