# SmartAttend – Model Card

**Last updated:** 2026-10-14  
Status entries: `SELECTED` · `FEASIBLE-PENDING` · `FALLBACK` · `REVIEWED/NOT-IMPLEMENTED` · `BLOCKED`

---

## Face Recognition

### Candidate A: FaceLiVT (2025) — FEASIBLE-PENDING

| Field | Value |
|---|---|
| Paper | FaceLiVT: Face Recognition Using Linear Vision Transformer with Structural Reparameterization for Mobile Device |
| Venue | IEEE ICIP 2025, pp. 1720–1725 |
| DOI | https://doi.org/10.1109/ICIP55913.2025.11084611 |
| Official repo | https://github.com/novendrastywn/FaceLiVT |
| License | **MIT** |
| Checkpoint variants | `facelivt_s`, `facelivt_m`, `facelivtv2_xs` (2.9 M params), `facelivtv2_s`, `facelivtv2_m`, `facelivtv2_l` |
| Trained on | Glint360K |
| CPU inference | Designed for mobile/edge; `map_location='cpu'` documented |
| Dependencies | torch, torchvision, timm, opencv-python, face_alignment, numpy |
| FaceLiVTv2 status | Listed as "under review" in repo — do NOT cite as peer-reviewed without verifying |
| Weights download | Referenced in README; **must confirm links are live before use** |
| Status | **FEASIBLE-PENDING**: checkpoint download not yet attempted; install not yet verified |
| Action | Phase 3: clone repo, attempt checkpoint download, run CPU load test |

### Candidate B: EdgeFace (2024) — FALLBACK / LICENSE CAUTION

| Field | Value |
|---|---|
| Paper | EdgeFace: Efficient Face Recognition Model for Edge Devices |
| Venue | IEEE Trans. Biometrics, Behavior & Identity Science, 6(2), 158–168 (2024) |
| DOI | https://doi.org/10.1109/TBIOM.2024.3352164 |
| Official repo | https://github.com/otroshi/edgeface |
| License | **CC BY-NC-SA 4.0 — Non-Commercial only** ⚠️ |
| Checkpoint variants | `edgeface_base`, `edgeface_s_gamma_05`, `edgeface_xs_gamma_06`, `edgeface_xxs`, quantized variants |
| Load method | `torch.hub.load('otroshi/edgeface', 'edgeface_xs_gamma_06', pretrained=True)` |
| CPU inference | Confirmed: no CUDA required |
| Dependencies | torch, torchvision, timm, mxnet (⚠️ large), opencv-python |
| Status | **FALLBACK**: license limits commercial use; mxnet dependency may cause friction on Windows CPU |
| Action | Phase 3: test install without mxnet first; confirm license applies to research prototype |

---

## Face Detection

### Candidate: YOLO-FaceV2 (2024) — FEASIBLE-PENDING

| Field | Value |
|---|---|
| Paper | YOLO-FaceV2: A Scale and Occlusion Aware Face Detector |
| Venue | Pattern Recognition, 155, article 110714 (2024) |
| DOI | https://doi.org/10.1016/j.patcog.2024.110714 |
| Official repo | https://github.com/Krasjet-Yu/YOLO-FaceV2 |
| License | MIT |
| Checkpoint variants | `yolo-facev2n.pt`, `yolo-facev2s.pt`, `yolo-facev2m.pt`, `yolo-facev2l.pt` |
| CPU inference | Supports `--device cpu` (YOLOv5-based) |
| Dependencies | torch, torchvision, opencv-python, scipy, PyYAML, thop, tqdm, pandas, matplotlib |
| Weights availability | ⚠️ Not confirmed directly downloadable — must check Releases tab and Issues |
| Status | **FEASIBLE-PENDING**: weights availability unconfirmed |
| Fallback if blocked | OpenCV Haar Cascade (built-in) or MediaPipe FaceDetection — disclose if used |

### Disclosed Fallback: OpenCV Haar Cascade / MediaPipe

If YOLO-FaceV2 weights are not available or CPU latency is unacceptable:
- Use `cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')`
- Or MediaPipe Face Detection (BlazeFace, runs fast on CPU)
- **Must be clearly disclosed in the paper and UI diagnostics panel as the engineering fallback used**

---

## Presentation Attack Detection (PAD)

### Candidate A: KBS 2025 Spatio-Temporal PAD — REVIEWED / NOT IMPLEMENTED

| Field | Value |
|---|---|
| Paper | Spatio-temporal deep learning for improved face presentation attack detection |
| Venue | Knowledge-Based Systems, 311, article 113059 (2025) |
| DOI | https://doi.org/10.1016/j.knosys.2025.113059 |
| Authors | Shujaat Khan et al. |
| Official code | ❌ **NOT PUBLICLY AVAILABLE** |
| Status | **REVIEWED/NOT-IMPLEMENTED**: no public code or checkpoint found |
| Decision | Cannot implement; paper is citeable for related-work context only |

### Candidate B: CASO-PAD (Sci. Reports 2026) — FEASIBILITY-PENDING

| Field | Value |
|---|---|
| Paper | Face Presentation Attack Detection via Content-Adaptive Spatial Operators |
| Venue | Scientific Reports (Nature), September 2026 |
| DOI | https://doi.org/10.1038/s41598-026-67944-6 |
| Preprint | arXiv:2602.18965 |
| Author | Shujaat Khan |
| Official repo | https://github.com/Shujaat123/CASO-PAD |
| License | CC BY 4.0 (paper); repo license — must verify |
| Architecture | MobileNetV3 + involution; ~3.6 M params; ~0.64 GFLOPs at 256×256 |
| Pretrained weights | ⚠️ Repo exists; weights availability **not yet confirmed** |
| CPU inference | In principle yes (MobileNetV3, tiny model) |
| Status | **FEASIBILITY-PENDING**: must check repo Releases for weights |
| Action | Phase 5: clone repo, check for checkpoint, attempt CPU load |

> [!IMPORTANT]
> Until a real PAD model is verified and running, the UI **must** display `PAD_UNAVAILABLE`. No blink/head-turn challenge may be presented as equivalent to a validated PAD model.

---

## Decision rules (current state)

| Layer | Decision |
|---|---|
| Recognition | Await Phase 3 (FaceLiVT preferred; EdgeFace fallback subject to license confirmation) |
| Detection | Await Phase 2 (YOLO-FaceV2 preferred; Haar/MediaPipe if blocked) |
| PAD | `PAD_UNAVAILABLE` until CASO-PAD weights confirmed and loaded |

---

## What this card does NOT claim

- Published benchmark numbers from any paper are NOT SmartAttend results.
- "Mobile/edge" in a paper does not guarantee acceptable CPU latency on this specific i3 machine; latency must be measured.
- FaceLiVTv2's performance is "under review" — only peer-reviewed FaceLiVTv1 numbers may be cited (with the paper).
- CC BY-NC-SA models cannot be deployed in commercial products without a separate agreement.
