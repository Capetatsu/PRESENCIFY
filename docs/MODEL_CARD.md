# SmartAttend – Model Card & System Registry

**Last updated:** 2026-10-14  
**Target Environment:** Intel i3-class CPU (6 threads), 8 GB RAM, Windows 11, No CUDA / NVIDIA GPU

---

## 1. Face Detection

### Selected Engine: MediaPipe BlazeFace Short-Range (Tasks API)
- **Status:** `SELECTED & VERIFIED`
- **Model Asset:** `models/blaze_face_short_range.tflite` (229 KB)
- **License:** Apache 2.0
- **Input Resolution:** Dynamic (640x480 standard webcam BGR frame, converted internally to RGB SRGB)
- **Hardware Target:** CPU only (XNNPACK delegate)
- **Measured Latency:**
  - One-time load: **30.6 ms**
  - Average inference per frame (10 webcam frames): **3.5 ms** (peak: 6.2 ms)
- **Memory Overhead:** 0.70 MB peak RAM
- **Disclosure:** Engineering fallback chosen over YOLO-FaceV2 to guarantee <10 ms edge CPU latency without custom CUDA dependencies.

### Candidate Evaluated: YOLO-FaceV2 (Pattern Recognition, 2024)
- **Status:** `REVIEWED / REPLACED BY FALLBACK`
- **Paper:** *YOLO-FaceV2: A Scale and Occlusion Aware Face Detector*, DOI: [10.1016/j.patcog.2024.110714](https://doi.org/10.1016/j.patcog.2024.110714)
- **Repository:** `Krasjet-Yu/YOLO-FaceV2`
- **License:** MIT
- **Availability:** Releases tab contains `best.pt` and `preweight.pt` checkpoints.
- **Decision:** Requires custom patched YOLOv5 PyTorch dependency graph; higher CPU latency than BlazeFace on Intel i3 hardware. Disclosed as reviewed comparator.

---

## 2. Face Recognition

### Selected Model: EdgeFace XS gamma=0.6 (IEEE TBIOM 2024)
- **Status:** `SELECTED & VERIFIED`
- **Paper:** *EdgeFace: Efficient Face Recognition Model for Edge Devices*, IEEE TBIOM 6(2), 2024. DOI: [10.1109/TBIOM.2024.3352164](https://doi.org/10.1109/TBIOM.2024.3352164)
- **Repository:** `otroshi/edgeface` (Idiap Research Institute)
- **Checkpoint:** `edgeface_xs_gamma_06.pt` (7.17 MB)
- **License:** **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike 4.0 International)
  - Compliant with academic research prototype use.
- **Preprocessing:** 112x112 face crop, BGR to RGB, normalized via `(x / 127.5) - 1.0` to `[-1.0, 1.0]`, NCHW float32.
- **Output:** 512-dimensional L2-normalized embedding vector.
- **Measured Latency (Intel i3 CPU):**
  - One-time load: **224.7 ms**
  - Inference per face: **37.2 ms average** (peak 67.6 ms)
- **Decision Thresholds (Calibrated Starting Defaults):**
  - Cosine similarity $\ge 0.40 \rightarrow$ `PRESENT`
  - $0.30 \le \text{Cosine similarity} < 0.40 \rightarrow$ `REVIEW_REQUIRED`
  - $\text{Cosine similarity} < 0.30 \rightarrow$ `UNKNOWN`
- **Reporting Rule:** Cosine similarity is strictly documented as a similarity score in $[-1, 1]$ and **never** labeled as a probability.

### Candidate Evaluated: FaceLiVT (IEEE ICIP 2025)
- **Status:** `REVIEWED / WEIGHTS NOT RELEASED`
- **Paper:** *FaceLiVT: Face Recognition Using Linear Vision Transformer with Structural Reparameterization for Mobile Device*, IEEE ICIP 2025. DOI: [10.1109/ICIP55913.2025.11084611](https://doi.org/10.1109/ICIP55913.2025.11084611)
- **Repository:** `novendrastywn/FaceLiVT`
- **License:** MIT
- **Availability Audit:** Repository contains `<!-- TODO: Add download links for pretrained models -->` and 0 releases. No public model weights exist.
- **Decision:** Cannot be executed without weights. Documented as a reviewed candidate.

---

## 3. Presentation Attack Detection (PAD / Anti-Spoofing)

### Status: `PAD_UNAVAILABLE` (Truthfully Disclosed)

#### Candidate A: KBS 2025 Spatio-Temporal MobileNetV3 PAD
- **DOI:** [10.1016/j.knosys.2025.113059](https://doi.org/10.1016/j.knosys.2025.113059)
- **Status:** `REVIEWED / NO CODE RELEASED`
- **Audit:** Authors did not publish a code repository or model checkpoint.

#### Candidate B: CASO-PAD (Scientific Reports, September 2026)
- **DOI:** [10.1038/s41598-026-67944-6](https://doi.org/10.1038/s41598-026-67944-6)
- **Repository:** `Shujaat123/CASO-PAD`
- **Status:** `REVIEWED / WEIGHTS NOT RELEASED`
- **Audit:** Repository contains training routines (`train.py`, `build_model.py`), but has 0 releases and no pretrained checkpoint files.

#### Implementation Policy
- No unvalidated eye-blink or head-turning challenges are masqueraded as PAD.
- Live UI and experiment logs explicitly flag `pad_status: PAD_UNAVAILABLE`.
