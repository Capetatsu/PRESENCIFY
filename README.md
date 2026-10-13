# SmartAttend

**Research prototype** – AI-assisted classroom attendance using a webcam.  
Windows · CPU-only · Python 3.12 · No cloud dependencies.

> ⚠️ This is a research prototype with consenting participants. Not a production campus CCTV system.

---

## Quick start (Milestone 1 – verified)

```powershell
# 1. Clone / enter the project
cd D:\Things\Work\Projects\MiniProject\smartattend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Webcam smoke test (headless – no window)
python scripts/webcam_test.py --headless

# 4. Webcam live view (press q to quit)
python scripts/webcam_test.py

# 5. Run all tests
python -m pytest tests/ -v
```

## Verified (2026-10-14)

| Check | Result |
|---|---|
| Python 3.12.1 | ✅ |
| PyTorch 2.14.0+cpu | ✅ |
| OpenCV 4.10.0 | ✅ |
| Camera at index 0 (DirectShow) | ✅ 640×480 BGR |
| `python scripts/webcam_test.py --headless` | ✅ PASS |
| `pytest tests/test_webcam.py` | ✅ 7/7 PASS |

## Still to install

```powershell
pip install streamlit>=1.38.0 onnxruntime>=1.19.0
```

## Project structure

```
smartattend/
├── docs/
│   ├── IMPLEMENTATION_STATUS.md   # milestone checklist and risks
│   ├── MODEL_CARD.md              # verified model info, licenses, status
│   └── SMOKETEST_PLAN.md         # isolated CPU smoke-test plan per model
├── experiments/                   # anonymized CSV, metrics, manifests
├── models/                        # checkpoint files (not in git)
├── scripts/
│   ├── webcam_test.py             # Milestone 1 – webcam smoke test
│   └── smoke/                     # per-model isolated smoke tests (Phase 2–5)
├── src/smartattend/               # main application package (Phase 4+)
├── tests/
│   └── test_webcam.py             # 7 pytest cases (all pass)
├── requirements.txt
└── pyproject.toml
```

## Research context

**Question:** On a CPU webcam prototype, does adding PAD and confidence-based review to a recent lightweight face-recognition pipeline reduce incorrect automatic attendance decisions compared with recognition-only, under a documented controlled test protocol?

This is a hypothesis — not a result or a novelty claim.

See [`SmartAttend_Agent_Pack/`](../SmartAttend_Agent_Pack/) for full project context, literature review, and agent rules.
