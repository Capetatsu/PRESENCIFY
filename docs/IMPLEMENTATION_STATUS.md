# SmartAttend – Implementation Status

**Last updated:** 2026-10-14  
**Branch:** `main`  
**Machine:** Windows, Intel i3-class CPU, 8 GB RAM, Intel UHD, no NVIDIA GPU, Python 3.12.1

---

## Current environment (verified)

| Item | Verified value |
|---|---|
| Python | 3.12.1 (C:\Program Files\Python312\python.exe) |
| PyTorch | 2.14.0+cpu (CPU-only build confirmed) |
| CUDA available | False |
| OpenCV | 4.10.0.84 |
| NumPy | 2.0.0 |
| Pandas | 2.3.3 |
| SQLAlchemy | 2.0.25 |
| pytest | 7.4.4 |
| Streamlit | ❌ NOT installed |
| onnxruntime | ❌ NOT installed |
| Camera | Index 0 opens via DirectShow; reads 640×480 BGR uint8 frames |

---

## Phase checklist

### ✅ Phase 0: Repository inspection
- [x] No prior codebase existed; project folder contained only the agent-pack markdown docs
- [x] No git repository existed (initialised fresh in `smartattend/`)
- [x] All 6 agent-pack markdown files read
- [x] `docs/IMPLEMENTATION_STATUS.md` created (this file)
- [x] `docs/MODEL_CARD.md` created

### ✅ Phase 1: Camera proof (Milestone 1 – COMPLETE)
- [x] `scripts/webcam_test.py` created
- [x] Headless mode: PASS (`--headless` exits 0)
- [x] `tests/test_webcam.py`: **7/7 tests PASS**
- [x] Camera opens cleanly on DirectShow, MSMF and CAP_ANY
- [x] Camera releases cleanly (double open/close cycle passes)
- [x] Invalid camera index raises `RuntimeError` (not a crash)
- Run command: `python scripts/webcam_test.py --headless`
- Live view command: `python scripts/webcam_test.py`
- Test command: `python -m pytest tests/test_webcam.py -v`

### 🔲 Phase 2: Face detector
- [ ] YOLO-FaceV2 feasibility gate (see model card)
- [ ] Fallback detector identified
- [ ] Smoke test: still image + webcam frame
- [ ] No-face case handled
- [ ] CPU latency logged

### 🔲 Phase 3: Recognition model feasibility gate
- [ ] FaceLiVT checkpoint download and CPU load verified
- [ ] EdgeFace checkpoint download and CPU load verified (license: CC BY-NC-SA 4.0 – non-commercial only)
- [ ] Embedding shape and determinism verified
- [ ] Per-face latency and RAM measured

### 🔲 Phase 4: Enrollment and recognition
- [ ] Student enrollment flow with consent flag
- [ ] Template generation and storage
- [ ] Threshold-based matching (similarity, not probability)
- [ ] Unknown face → UNKNOWN; ambiguous → REVIEW_REQUIRED

### 🔲 Phase 5: PAD feasibility gate
- [ ] CASO-PAD (Sci. Reports 2026) code/weights verified
- [ ] KBS 2025 PAD – code not publicly available (marked: reviewed candidate, NOT implemented)
- [ ] UI truthfully reports PAD_UNAVAILABLE until a real model is running

### 🔲 Phase 6: Attendance persistence (SQLite)
- [ ] Schema: students, face_templates, sessions, attendance_events, review_events
- [ ] Unique constraint: accepted (student_id, session_id)
- [ ] Session start/end, duplicate prevention, manual review/audit events

### 🔲 Phase 7: Streamlit dashboard
- [ ] Install streamlit
- [ ] Pages: dashboard, enroll, live attendance, review queue, history, export, diagnostics

### 🔲 Phase 8: Experiments and paper
- [ ] Experiment mode: pseudonymous CSV export
- [ ] Metrics computed from raw logs (not hardcoded)
- [ ] Literature matrix complete

### 🔲 Phase 9: Final checks and delivery

---

## Risks and blockers

| Risk | Severity | Status |
|---|---|---|
| **YOLO-FaceV2 weights availability** – pretrained `.pt` files not confirmed directly downloadable from repo | HIGH | Not yet checked directly; must visit Releases tab before Phase 2 |
| **FaceLiVT checkpoint download** – links referenced in README but not confirmed live | HIGH | Pending direct download attempt |
| **EdgeFace license: CC BY-NC-SA 4.0** – non-commercial only | MEDIUM | Must confirm SmartAttend qualifies as non-commercial research; if not, use FaceLiVT (MIT) only |
| **EdgeFace depends on mxnet** – large/fragile dependency on Windows CPU | MEDIUM | Test install separately before committing |
| **KBS 2025 PAD – no public code** | MEDIUM | Cannot implement; marked reviewed/not-implemented; paper still citeable for context |
| **CASO-PAD weights** – code repo exists, weights availability unconfirmed | MEDIUM | Must check repo Releases tab directly |
| **Streamlit not installed** | LOW | Install: `pip install streamlit>=1.38.0` |
| **onnxruntime not installed** | LOW | Install: `pip install onnxruntime>=1.19.0` |
| **Python 3.12 vs spec (3.11)** | LOW | 3.12 is installed; all current packages work; note the deviation from spec |
| **Paper deadline: 15 October 2026** | CRITICAL | Today is 14 October 2026 – one day remaining; scope must be tightly controlled |

> [!CAUTION]
> **ONE DAY TO DEADLINE (15 Oct 2026).** Priority must be: Phase 2 (detector) → Phase 3 (recognition model working on CPU) → Phase 4 (enrollment + matching) → minimal Phase 7 (Streamlit UI). Phases 5, 6, 8, 9 must be scoped to what is achievable today.

---

## Next milestone (Phase 2)

1. Attempt to download `yolo-facev2s.pt` from the YOLO-FaceV2 repo Releases  
2. If unavailable: use `opencv.CascadeClassifier` (Haar) or `mediapipe` FaceDetection as a disclosed engineering fallback  
3. Run detector smoke test on a still image and a webcam frame  
4. Log CPU latency; if >500 ms/frame choose a lighter detector  
