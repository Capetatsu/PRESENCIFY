# SmartAttend – CPU Smoke-Test Plan

**Machine target:** Intel i3-class CPU, 8 GB RAM, Intel UHD, no CUDA, Windows, Python 3.12.1  
**Purpose:** Isolated verification of each model candidate before integration.  
**Rule:** Do not install a research repo's full dependency tree blindly. Test in isolation first.

---

## Smoke-test protocol (applies to every candidate)

For each model, run a standalone Python script that:

1. **Imports** only the packages needed (no global installs of research repos).
2. **Loads the checkpoint** to CPU (`map_location='cpu'`); records load time.
3. **Runs one forward pass** on a synthetic or consented test image of the correct input shape.
4. **Checks** output shape, value range and determinism (two passes must produce the same output).
5. **Measures** per-inference latency (average of 5 runs after 1 warm-up).
6. **Reports** peak RAM (using `tracemalloc` or `psutil`).
7. **Exits with code 0** if all checks pass, non-zero otherwise.
8. **Logs** exact checkpoint filename/hash, model class, device, latency and any error.

All scripts live in `scripts/smoke/`. They are standalone; the main app does not import them.

---

## Plan by component

### 1. Face Detector: YOLO-FaceV2 (MIT, 2024)

**Repo:** https://github.com/Krasjet-Yu/YOLO-FaceV2  
**Script:** `scripts/smoke/smoke_detector_yolov2face.py`

| Step | Detail |
|---|---|
| Pre-check | Download `yolo-facev2s.pt` from Releases tab; record URL and SHA-256 |
| Input | 640×640 RGB image (nearest multiple); test image: `tests/fixtures/face_frontal.jpg` |
| Output | List of bounding boxes and confidences |
| Pass criteria | ≥1 face detected on frontal test image; no-face image returns empty; latency <500 ms |
| Fallback trigger | Weights not available OR latency >500 ms/frame → use MediaPipe/Haar fallback |
| Disclosure | If fallback is used, record it in MODEL_CARD.md and diagnostics panel |

**Fallback (if YOLO-FaceV2 blocked):**  
`cv2.CascadeClassifier` (built-in, no download needed) or `mediapipe>=0.10` FaceDetection.  
Script: `scripts/smoke/smoke_detector_haar.py`

---

### 2. Recognition Candidate A: FaceLiVT (MIT, IEEE ICIP 2025)

**Repo:** https://github.com/novendrastywn/FaceLiVT  
**Script:** `scripts/smoke/smoke_recognition_facelivt.py`

| Step | Detail |
|---|---|
| Pre-check | Clone repo; confirm checkpoint link in README is live; download `facelivt_s` or `facelivtv2_xs` |
| Dependencies | `pip install timm face_alignment` (timm is already compatible; face_alignment needed for alignment) |
| Preprocessing | 112×112 aligned face crop, normalize to [-1,1] or per-paper normalization |
| Input shape | (1, 3, 112, 112) float32 |
| Output | 512-d or paper-specified embedding vector |
| Pass criteria | Checkpoint loads to CPU without CUDA error; output shape correct; two runs identical (deterministic); load time <30 s; per-face latency <500 ms |
| Notes | FaceLiVTv2 is "under review" — use peer-reviewed v1 checkpoint for the paper |

---

### 3. Recognition Candidate B: EdgeFace (CC BY-NC-SA 4.0, IEEE TBIOM 2024)

**Repo:** https://github.com/otroshi/edgeface  
**Script:** `scripts/smoke/smoke_recognition_edgeface.py`

| Step | Detail |
|---|---|
| License gate | Confirm SmartAttend qualifies as non-commercial research before using |
| Load method | `torch.hub.load('otroshi/edgeface', 'edgeface_xs_gamma_06', source='github', pretrained=True)` — no mxnet needed for inference |
| Input shape | 112×112 aligned face crop |
| Output | 512-d embedding |
| Pass criteria | Loads without mxnet; embedding shape (512,); deterministic; latency <300 ms |
| Notes | If `mxnet` is pulled in as a transitive dep, attempt `pip install edgeface --no-deps` workaround |

---

### 4. PAD Candidate A: KBS 2025 Spatio-Temporal PAD

**Status: REVIEWED / NOT IMPLEMENTED**  
No public code or checkpoint found. Skip smoke test; note in MODEL_CARD.md.

---

### 5. PAD Candidate B: CASO-PAD (CC BY 4.0, Sci. Reports 2026)

**Repo:** https://github.com/Shujaat123/CASO-PAD  
**Script:** `scripts/smoke/smoke_pad_casoped.py`

| Step | Detail |
|---|---|
| Pre-check | Clone repo; check Releases for checkpoint; record filename and SHA-256 |
| Architecture | MobileNetV3 + involution; ~3.6 M params |
| Input | 256×256 face crop (single RGB frame) |
| Output | Scalar score or binary logit (genuine/spoof) |
| Pass criteria | Model loads on CPU; single-frame inference <200 ms; scores differ between a frontal face image and a printed-photo image |
| Notes | Paper trained on Replay-Attack, Replay-Mobile, ROSE-Youtu, OULU-NPU, SiW-Mv2 — do not transfer those benchmark metrics to SmartAttend results |
| Fallback trigger | Weights unavailable → UI must show `PAD_UNAVAILABLE`; do not fake PAD |

---

## Test fixtures needed

| File | Purpose |
|---|---|
| `tests/fixtures/face_frontal.jpg` | Consented still face image, frontal, good lighting |
| `tests/fixtures/no_face.jpg` | Image with no face (e.g., desk or wall) |
| `tests/fixtures/printed_photo.jpg` | Printed A4 photo held to camera (PAD attack test) |

> [!IMPORTANT]
> All test fixtures must use consenting participants or synthetic faces (e.g., DALL-E generated). Never use scraped real faces without consent.

---

## Acceptance criteria summary

| Model | Load OK | Forward OK | Latency target | Deterministic | Decision |
|---|---|---|---|---|---|
| YOLO-FaceV2 or fallback detector | ☐ | ☐ | <500 ms/frame | ☐ | PENDING |
| FaceLiVT (preferred) | ☐ | ☐ | <500 ms/face | ☐ | PENDING |
| EdgeFace (fallback) | ☐ | ☐ | <300 ms/face | ☐ | PENDING |
| KBS 2025 PAD | N/A | N/A | N/A | N/A | NOT IMPLEMENTED |
| CASO-PAD | ☐ | ☐ | <200 ms/frame | ☐ | PENDING |

Fill in checkboxes during Phase 2–5 smoke tests. Do not check a box until the script actually passes.
