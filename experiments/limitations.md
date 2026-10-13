# SmartAttend – Limitations & Scope Disclosures

**Date:** October 2026  
**Document:** `experiments/limitations.md`

---

## 1. Prototype and Hardware Scope

1. **Local Prototype vs Campus CCTV Infrastructure:**
   - SmartAttend is explicitly evaluated as a lightweight, single-workstation prototype running on entry-level laptop hardware (Intel i3-class CPU, 8 GB RAM, Intel UHD Graphics).
   - It is designed for interactive check-in / session-based desk kiosks, **not** for wide-angle, multi-camera campus-wide surveillance.
   - Findings reflect laptop webcam optics (640×480 BGR) and near-field participant distance (~0.5 m to 1.5 m).

2. **Inference Latency Context:**
   - Feature extraction with EdgeFace XS achieved an average latency of **21.31 ms** (median 20.03 ms) on the target Intel i3 CPU.
   - MediaPipe BlazeFace detection executes in **3.5 ms**.
   - These measurements represent CPU inference under standard thread pools without hardware GPU acceleration.

---

## 2. Face Recognition & Demographic Generalizability

1. **Pilot Evaluation Scale:**
   - The controlled benchmark evaluation was conducted on a pilot cohort of 41 subjects (410 probe captures).
   - Results demonstrate the structural benefit of dual-threshold confidence routing over a hard cutoff, but do not claim universal generalizability across thousands of students or diverse worldwide demographic distributions.
2. **Environmental Sensitivity:**
   - Recognition accuracy decreases under severe side-profile angles (>45° yaw) or extreme illumination drops (<15 lux).
   - In such conditions, similarity scores consistently fall below $\tau_{\text{accept}} = 0.65$, effectively triggering human review rather than silent auto-admission.

---

## 3. Presentation Attack Detection (PAD) Disclosures

1. **Current PAD State: UNAVAILABLE:**
   - No validated deep-learning anti-spoofing model is active in this prototype.
   - The UI and logging pipelines truthfully label `pad_status: PAD_UNAVAILABLE`.
   - The system **does not** prevent presentation attacks (e.g., printed photo presentations, 2D screen replays, or physical masks).
2. **Evaluated PAD Candidates:**
   - *KBS 2025 Spatio-Temporal MobileNetV3 PAD*: No runnable code or model weights were publicly released by the authors.
   - *CASO-PAD (Scientific Reports, September 2026)*: Repository `Shujaat123/CASO-PAD` provides training scripts, but provides zero public model releases or checkpoint downloads.
   - SmartAttend refuses to substitute heuristic eye-blink or head-turning challenges as "anti-spoofing", as literature (Shaheed et al., 2024; Pooshideh et al., 2025) demonstrates that simple interactive heuristics are easily bypassed and do not constitute validated PAD.

---

## 4. Privacy and Biometric Ethics

1. **Local-Only Storage:**
   - All face templates and student records reside in a local SQLite database (`smartattend.db`).
   - No biometric data, images, or embeddings are transmitted to cloud endpoints.
2. **Informed Consent Required:**
   - Students cannot be enrolled without affirmative consent recording in the enrollment schema.
3. **Template Privacy:**
   - In accordance with findings by Sun and Liu (2025), raw face embeddings are not treated as "anonymous" tokens, as template inversion attacks remain theoretically possible. Template deletion is directly supported.
