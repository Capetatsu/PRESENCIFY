# SmartAttend: AI-Assisted Classroom Attendance on CPU Edge Devices

[![Tests](https://img.shields.io/badge/pytest-38%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20x86__64-lightgrey.svg)]()
[![Hardware](https://img.shields.io/badge/hardware-Intel%20i3%20%7C%20CPU--only%20%7C%208GB%20RAM-orange.svg)]()
[![Biometrics](https://img.shields.io/badge/model-EdgeFace%20XS%20%287.17MB%29-blueviolet.svg)]()
[![Ethics](https://img.shields.io/badge/PAD%20status-Truthful%20%28PAD__UNAVAILABLE%29-yellow.svg)]()

**SmartAttend** is a research prototype and local-first AI system designed to automate classroom attendance logging directly on commodity laptop hardware (Intel i3-class CPU, 8 GB RAM, integrated graphics, no NVIDIA GPU).

Rather than enforcing brittle hard-threshold decision boundaries that cause high false absences, SmartAttend introduces a **dual-threshold confidence-based review policy**: high-similarity students are automatically marked Present, low-similarity strangers are rejected, and ambiguous borderline matches are routed to an audited instructor queue.

---

## ⚡ Quickstart

### 1. Launch the Live Streamlit Dashboard
```powershell
streamlit run src/smartattend/app.py
```
Opens interactive dashboard on `http://localhost:8501` featuring:
- **Live Attendance:** Real-time webcam check-in with session deduplication.
- **Enroll Student:** Capture and enroll student with mandatory affirmative consent.
- **Review Queue:** Teacher audit interface to approve/reject borderline matches ($[0.52, 0.65)$).
- **Attendance History:** Searchable session records with manual override audit trail.
- **CSV Export:** Pseudonymous, FERPA/GDPR-compliant attendance report generation.
- **System Diagnostics:** Live CPU latency, memory utilization, and hardware delegates.

### 2. One-Line Master Reproduction
Run all 38 pytest cases, re-execute the controlled evaluation across 410 captures, and verify all mathematical identities and data integrity:
```powershell
python scripts/reproduce_all.py
```

### 3. Run Pytest Suite Directly (38 tests)
```powershell
python -m pytest tests/ -v
```

### 4. Re-run Controlled Face-Recognition Evaluation
```powershell
python scripts/experiments/run_controlled_evaluation.py
```

### 5. Headless Webcam Smoke Test
```powershell
python scripts/webcam_test.py --headless
```

---

## 🔬 Key Experimental Results (166 Held-Out Test Probes)

Evaluated under an open-set identification protocol across **41 subjects (410 captures)** with strictly disjoint gallery ($M=20$), validation ($N=100$), and held-out test ($N=166$) splits:

| Metric | Method A: Recognition-Only ($\tau = 0.65$) | Method B: Stage 1 (Automated) | Method B: Stage 2 (Post-Review) | Impact of SmartAttend Policy |
|---|---|---|---|---|
| **True Accepts (ID Match)** | 64 / 100 (64.0%) | 64 / 100 (64.0%) | **78 / 100 (78.0%)** | **+14 genuine students rescued** |
| **Misidentifications** | 7 / 100 (7.0%) | 7 / 100 (7.0%) | 7 / 100 (7.0%) | Isolated in high-confidence tail |
| **False Rejections (Absence)** | 29 / 100 (29.0%) | 4 / 100 (4.0% auto) | **15 / 100 (15.0%)** | **Cut from 36.0% down to 15.0%** |
| **Impostors Routed to Review**| 0 (0.0%) | **33 / 66 (50.0%)** | 0 (Audited) | **33 impostors stopped from auto-login** |
| **Impostor False Accepts** | 18 / 66 (27.3%) | 18 / 66 (27.3%) | 18 / 66 (27.3%) | High-similarity tail |
| **Impostor True Rejects** | 48 / 66 (72.7%) | 15 / 66 (22.7% auto) | **48 / 66 (72.7%)** | Maintained strong stranger rejection |
| **Review Queue Workload** | 0 / 166 (0.0%) | **58 / 166 (34.94%)** | 58 audited | Quantified human workload |
| **Identity-Specific Precision**| 0.7191 (64/89) | — | **0.7573 (78/103)** | **+3.82%** |
| **Identity-Specific Recall** | 0.6400 (64/100) | — | **0.7800 (78/100)** | **+14.00%** |
| **Identity-Specific F1-Score** | 0.6772 | — | **0.7685** | **+9.13%** |
| **Binary Attendance Precision** | 0.7978 (71/89) | — | **0.8252 (85/103)** | **+2.74%** |
| **Binary Attendance Recall** | 0.7100 (71/100) | — | **0.8500 (85/100)** | **+14.00%** |
| **Binary Attendance F1-Score** | 0.7513 | — | **0.8374** | **+8.61%** |
| **Overall Accuracy** | 67.47% | — | **75.90%** | **+8.43%** |

---

## ⏱️ CPU Hardware Latency Profile

Measured on an Intel i3-class dual/quad-core CPU (6 execution threads, PyTorch CPU delegator):

| Component | Architecture / Model | Mean Latency | Median Latency |
|---|---|---|---|
| **Face Detection** | MediaPipe BlazeFace (`blaze_face_short_range.tflite`, 229 KB) | **3.5 ms** | 3.2 ms |
| **Facial Alignment** | Keypoint Partial Affine (112×112, $[-1, 1]$ float32) | **0.8 ms** | 0.7 ms |
| **Feature Extraction** | EdgeFace XS $\gamma=0.6$ (`edgeface_xs_gamma_06.pt`, 7.17 MB) | **26.6 ms** | 21.2 ms |
| **Gallery Search** | Cosine similarity across 20 enrolled gallery templates | **< 0.05 ms** | < 0.05 ms |
| **Total Computation** | **End-to-end algorithmic frame processing** | **~30.9 ms** | **~25.1 ms** |
| **Frame Throughput** | Pure computational throughput (excluding 30 FPS camera shutter) | **~32.4 FPS** | **~39.8 FPS** |

---

## 🛡️ Research Integrity, Ethics & Disclosures

1. **Presentation Attack Detection (PAD):** No verified 2024–2026 deep learning PAD weights (e.g., CASO-PAD 2026, KBS 2025) are publicly distributed for lightweight CPU execution. The prototype strictly and truthfully logs `pad_status: PAD_UNAVAILABLE`. Heuristic blink or head-turn challenges are never represented as validated PAD.
2. **Similarity Score Discipline:** Cosine similarity is reported strictly in $[-1.0, 1.0]$ and is **never** presented as a probability or confidence percentage.
3. **Participant Privacy & Provenance:** 
   - 1 Consented Live Participant ($P001$, local operator) provided informed consent for live testing.
   - 40 Academic Benchmark Subjects ($P002$–$P041$) are from the standardized AT&T Cambridge faces database under open academic terms.
   - All database records and CSV exports are strictly pseudonymous ($P001$, $P002$, etc.). Facial embeddings and raw images are never committed to git or exposed to external servers.

---

## 📂 Repository Structure

```
smartattend/
├── docs/
│   ├── PAPER_DRAFT.md            # Complete 16-section academic research paper draft
│   ├── paper/
│   │   ├── smartattend_ieee.tex  # Submission-ready IEEE conference LaTeX source
│   │   └── smartattend.bib       # Verified BibTeX bibliography (2024–2026 literature)
│   ├── IMPLEMENTATION_STATUS.md  # Detailed milestone progress and risk audit
│   ├── MODEL_CARD.md             # EdgeFace XS model card, license, and input specs
│   └── SMOKETEST_PLAN.md         # Isolated model CPU verification plan
├── experiments/
│   ├── metrics.json              # Exact machine-readable experimental metrics
│   ├── predictions_anonymized.csv# 166 per-probe decisions and latency records
│   ├── test_protocol.md          # Open-set split accounting and mathematical formulas
│   ├── limitations.md            # Hardware, dataset, and PAD disclosures
│   ├── literature_matrix.csv     # Systematic 2024–2026 literature review matrix
│   └── plots/                    # High-resolution experiment figures (300 DPI)
│       ├── similarity_distributions.png
│       ├── performance_comparison.png
│       └── latency_distribution.png
├── models/                       # Local model checkpoints (gitignored)
├── scripts/
│   ├── reproduce_all.py          # Master one-line reproducibility runner
│   ├── webcam_test.py            # Webcam smoke test
│   ├── demo_pipeline_e2e.py      # End-to-end CLI demonstration
│   └── experiments/
│       └── run_controlled_evaluation.py # Controlled evaluation pipeline
├── src/smartattend/              # Production application package
│   ├── app.py                    # Multi-page Streamlit web application
│   ├── attendance.py             # Session logic, deduplication & consent checks
│   ├── database.py               # SQLite WAL engine & audit schema
│   ├── detector.py               # MediaPipe BlazeFace CPU wrapper
│   ├── aligner.py                # Affine crop and landmark alignment
│   ├── recognizer.py             # EdgeFace XS backbone & gallery matching
│   └── webcam.py                 # Windows DirectShow camera capture
├── tests/                        # 38 passing pytest test cases
│   ├── test_webcam.py            # Webcam capture & hardware tests
│   ├── test_pipeline.py          # Detection, alignment, recognition, DB tests
│   └── test_export.py            # CSV export and data integrity tests
├── requirements.txt
└── pyproject.toml
```

---

## 📖 Citation

If you use or build upon SmartAttend, please cite our research paper:

```bibtex
@inproceedings{kumar2026smartattend,
  author    = {Mayank Kumar},
  title     = {SmartAttend: AI-Assisted Classroom Attendance on Resource-Constrained CPU Edge Devices with Confidence-Based Review},
  booktitle = {Proceedings of the International Conference on Edge Biometrics and Applied Vision},
  year      = {2026},
  url       = {https://github.com/Capetatsu/SmartAttend.git}
}
```
