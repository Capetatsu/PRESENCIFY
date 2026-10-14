# PRESENCIFY – Implementation Status

**Last updated:** 2026-10-14  
**Machine:** Windows 11, Intel i3-class CPU, 8 GB RAM, Intel UHD Graphics, no CUDA/NVIDIA GPU, Python 3.12.1  
**Repository:** https://github.com/Capetatsu/PRESENCIFY.git (branch `main`)

---

## Current Environment (Verified)

| Component | Status / Version | Notes |
|---|---|---|
| Python | 3.12.1 (64-bit) | Standard Windows CPython |
| PyTorch | 2.14.0+cpu | Verified `torch.cuda.is_available() == False`, 6 CPU threads |
| OpenCV | 4.10.0.84 | Verified DirectShow, MSMF, CAP_ANY (640x480 uint8) |
| MediaPipe | 1.1.0 | Tasks API (`blaze_face_short_range.tflite`, 229 KB). CPU latency: 3.5 ms |
| EdgeFace | XS gamma=0.6 (`edgeface_xs_gamma_06.pt`, 7.17 MB) | IEEE TBIOM 2024. Forward pass CPU latency: 26.6 ms mean |
| Streamlit | 1.40.1 | Full 6-page dashboard operational on CPU |
| SQLite / SQLAlchemy | 2.0.25 | WAL mode enabled, duplicate prevention, foreign keys |
| pytest | 7.4.4 | 38/38 tests passing |

---

## Phase Checklist & Status

### [x] Phase 0: Repository Inspection & Scaffolding
- [x] Initial inspection completed. No preexisting code found; initialized repository with clean directory hierarchy.
- [x] Formulated test plans, model cards, and documentation.

### [x] Phase 1: Camera Proof & Baseline Smoke Test
- [x] Verified Windows webcam capture using DirectShow fallback pipeline.
- [x] Created `scripts/webcam_test.py` (headless & live window).
- [x] 7/7 tests passing in `tests/test_webcam.py`.

### [x] Phase 2: Face Detector Verification (MediaPipe vs YOLO-FaceV2)
- [x] **MediaPipe FaceDetection (BlazeFace)**:
  - Installed `mediapipe==1.1.0`. Downloaded verified model `blaze_face_short_range.tflite` (229 KB, Apache 2.0).
  - Isolated CPU smoke test passed (`scripts/smoke/smoke_detector_mediapipe.py`).
  - Tested on synthetic image: 1 face detected, 0 detections on blank image.
  - Tested on live webcam: 10 frames detected 1 face consistently.
  - **Latency**: 30.6 ms one-time model load; **3.5 ms average CPU inference** per frame!
  - **Memory**: 0.70 MB peak RAM.
- [x] **YOLO-FaceV2 Check**:
  - Investigated GitHub repository `Krasjet-Yu/YOLO-FaceV2`.
  - Releases v1.0 and v2.1 contain initial/preweights (`best.pt`, `yolo-facev2s-preweight.pt`).
  - However, requires heavy custom YOLOv5 dependencies and PyYAML patching, whereas MediaPipe BlazeFace executes in 3.5 ms with zero GPU overhead.
  - MediaPipe is selected and fully disclosed as the verified detector engine.

### [x] Phase 3: Face Recognition Model Feasibility Gate
- [x] **FaceLiVT (IEEE ICIP 2025) Audit**:
  - Examined repository `novendrastywn/FaceLiVT`.
  - Inspected official GitHub repository and README.
  - **BLOCKER VERIFIED**: Repository contains `<!-- TODO: Add download links for pretrained models -->` and 0 releases. No public pretrained checkpoints exist.
  - Status: Marked as `REVIEWED / WEIGHTS NOT RELEASED`. Not selectable.
- [x] **EdgeFace (IEEE TBIOM 2024) Verification**:
  - Idiap Research Institute, CC BY-NC-SA 4.0 (compliant for academic research prototype).
  - Bundled checkpoint `edgeface_xs_gamma_06.pt` (7.17 MB) loaded and verified via CPU PyTorch.
  - 512-dimensional L2-normalized embedding generated. Finite, deterministic output verified.
  - **Latency**: 387 ms one-time model load; **26.6 ms average CPU latency** forward pass.
  - Selected as primary face recognition model.

### [x] Phase 4: Enrollment and Recognition Engine
- [x] Consent confirmation required before student record creation.
- [x] 112x112 face alignment and ArcFace-style `[-1, 1]` normalization in `aligner.py`.
- [x] Template extraction and storage in SQLite with model version tracking.
- [x] Calibrated thresholds: $\tau_{\text{accept}} = 0.65$, $\tau_{\text{review}} = 0.52$.
- [x] Cosine similarity strictly reported as a similarity score, never as a probability.

### [x] Phase 5: Presentation Attack Detection (PAD) Feasibility Gate
- [x] **KBS 2025 Spatio-Temporal PAD**: No public code found. Status: `NOT IMPLEMENTED`.
- [x] **CASO-PAD (Scientific Reports 2026)**:
  - Examined repository `Shujaat123/CASO-PAD`.
  - Found training/evaluation scripts, but 0 GitHub releases and no downloadable pretrained weights.
  - Status: `REVIEWED / WEIGHTS NOT RELEASED`.
- [x] **PAD Decision**: Truthfully configured as `PAD_UNAVAILABLE`. No fake liveness detection or unvalidated eye-blink tricks used.

### [x] Phase 6: Session Attendance, Duplicate Prevention, SQLite & Audit
- [x] SQLite database schema implemented (`students`, `face_templates`, `sessions`, `attendance_events`, `review_events`).
- [x] Database-level and application-level de-duplication: subsequent matches in the same session return `None` and do not record duplicate accepted attendance.
- [x] Teacher manual review implemented with full audit logging.
- [x] 38/38 unit and integration tests passing (`pytest tests/ -v`).

### [x] Phase 7: Streamlit Dashboard
- [x] Created `src/presencify/app.py` with 6 interactive pages:
  - Live Attendance (real-time webcam overlay with bounding boxes, similarity scores, and PAD banner)
  - Enroll Student (with consent confirmation and face template capture)
  - Review Queue (audited teacher resolution of uncertain matches)
  - Session History (session breakdown and accepted attendees)
  - Export (CSV and diagnostic downloads)
  - Diagnostics (real-time hardware, versions, latency, and model card inspection)

### [x] Phase 8: Pseudonymous Exports & Manifests
- [x] Created `src/presencify/export.py`:
  - `predictions_anonymized.csv`: exports pseudonyms (e.g. P001), similarity scores, and decisions without personal display names.
  - `attendance_summary.csv`: per-session aggregate metrics.
  - `environment.txt` & `model_manifest.json`: reproducibility manifests.

### [x] Phase 9: Controlled Evaluation & Integrity Audit
- [x] Evaluated across 41 subjects (410 captures total): 1 consented participant ($P001$) + 40 academic benchmark subjects ($P002$–$P041$).
- [x] Split accounting verified: 40 gallery + 100 validation + 166 held-out test + 104 unused = 410 captures (zero leakage).
- [x] Audited Method A vs Method B:
  - Rescues 14 genuine students from false rejection.
  - Intercepts 33 borderline impostors from auto-logging attendance.
  - Cuts false rejections from 36.0% down to 15.0%.
  - Improves identity F1-score from 0.677 to 0.769 (binary attendance F1 from 0.751 to 0.837).
- [x] Generated high-res plots (300 DPI) in `experiments/plots/`.

### [x] Phase 10: Research Paper & Master Reproduction
- [x] Drafted complete 16-section research paper: `docs/PAPER_DRAFT.md`.
- [x] Formatted IEEEtran two-column LaTeX submission package: `docs/paper/presencify_ieee.tex` and `docs/paper/presencify.bib`.
- [x] Authored one-line master reproduction script: `scripts/reproduce_all.py`.

---

## Summary of Verification Commands

| Action | Command | Result |
|---|---|---|
| Master Reproduction | `python scripts/reproduce_all.py` | ✅ All 4 steps pass with zero defects |
| Test Suite | `python -m pytest tests/ -v` | ✅ 38/38 passing (100%) |
| Controlled Evaluation | `python scripts/experiments/run_controlled_evaluation.py` | ✅ 166 probes evaluated, plots generated |
| Live Dashboard | `streamlit run src/presencify/app.py` | ✅ HTTP 200 on port 8501, 6 pages operational |
| Webcam Smoke Test | `python scripts/webcam_test.py --headless` | ✅ DirectShow capture verified |
