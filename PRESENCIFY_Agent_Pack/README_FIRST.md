# PRESENCIFY --- Agent Handoff Pack

**Deadline:** working demo + research paper before 15 October 2026.\
**Machine:** Windows, Intel i3-class CPU, 8 GB RAM, Intel UHD, no NVIDIA
GPU, laptop webcam.

## Read in order

1.  `PROJECT_CONTEXT.md` --- requirements and architecture.
2.  `RESEARCH_REVIEW_2024_2026.md` --- recent research candidates and
    literature notes.
3.  `IMPLEMENTATION_PLAN.md` --- staged implementation and model
    feasibility gates.
4.  `EXPERIMENTS_AND_PAPER.md` --- evaluation protocol and paper
    outline.
5.  `AGENT_RULES.md` --- rules for the coding agent.

The agent must inspect the existing repository before modifying it.
Build one testable milestone at a time. Never fabricate model results,
citations, tests or feature status.

**Research question candidate:** On a CPU-based webcam prototype, does
adding presentation-attack detection (PAD) and confidence-based review
to a recent lightweight face-recognition pipeline reduce incorrect
automatic attendance decisions compared with a recognition-only baseline
under a documented controlled test protocol?

This is a hypothesis, not a result or novelty claim.

**Model candidates:** FaceLiVT (2025) is the preferred recent
recognition research candidate if pretrained weights and CPU inference
can be verified. EdgeFace (published 2024) is a practical
comparison/fallback candidate with public code/checkpoints. YOLO-FaceV2
(2024) is a detector candidate subject to code/weights/CPU verification.
For PAD, evaluate the 2025 spatio-temporal MobileNetV3-based paper and
the 2026 CASO-PAD paper; do not claim either is implemented until code,
weights and preprocessing run successfully.

The final paper must accurately name the model and checkpoint that the
code actually runs.
