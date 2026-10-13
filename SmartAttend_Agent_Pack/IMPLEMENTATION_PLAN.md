# SmartAttend --- Implementation Plan

## Phase 0: inspect repository

Print the tree; inspect README, manifests, environment files, tests,
entry points and git status. Do not delete/rewrite files. Create
`docs/IMPLEMENTATION_STATUS.md` and `docs/MODEL_CARD.md`. Report current
state and first milestone before broad changes.

**Pass:** repository map and runnable commands are documented.

## Phase 1: camera proof

Create/verify a minimal OpenCV webcam test. Use DirectShow on Windows
with a fallback backend. Handle missing camera, stop cleanly and release
resources.

**Pass:** webcam feed opens and exits reliably.

## Phase 2: detector

Evaluate YOLO-FaceV2 (2024): verify implementation, weights, license and
CPU latency in an isolated smoke test. If it is infeasible, use the
smallest reliable detector as an engineering fallback and disclose it.

**Pass:** detector works on a still image and webcam frame; no-face case
works; CPU latency is logged.

## Phase 3: recognition model feasibility gate

### Candidate A: FaceLiVT (2025)

Repo: https://github.com/novendrastywn/FaceLiVT\
DOI: https://doi.org/10.1109/ICIP55913.2025.11084611

Verify checkpoint is actually available and licensed; inspect
alignment/normalization; load on CPU; check embedding shape, finite
values and deterministic inference; test a few consented samples; log
model load time, per-face latency and memory.

### Candidate B: EdgeFace (2024)

Repo: https://github.com/otroshi/edgeface\
DOI: https://doi.org/10.1109/TBIOM.2024.3352164

Use as a comparison or practical fallback if FaceLiVT is blocked. Verify
exact checkpoint, license, alignment and CPU inference.

**Feasibility report must include:** publication year/status, official
repo, checkpoint path/hash, license, preprocessing, CPU proof, latency,
memory and decision (selected/rejected/fallback). Do not spend the
deadline forcing a model without accessible weights.

## Phase 4: enrollment and recognition

Enroll unique student ID, name and consent. Capture multiple
good-quality images. Generate templates with selected model and store
model/preprocessing versions. Compare embeddings with a
validation-calibrated threshold and ambiguity margin. Similarity is not
probability. Unknown faces must not be assigned to the nearest enrolled
person without thresholding.

**Pass:** enrolled participant can be matched in controlled tests;
unknown person is unknown/reviewed; low-quality frames do not silently
create attendance.

## Phase 5: PAD feasibility gate

Candidates: - 2025 spatio-temporal PAD:
https://doi.org/10.1016/j.knosys.2025.113059 - 2026 CASO-PAD:
https://doi.org/10.1038/s41598-026-67944-6

Find official code/weights and license. Reproduce preprocessing and
temporal window if applicable. Test genuine face, printed photo and
screen replay separately. Log scores and threshold. If no recent model
is executable, label PAD unavailable or a clearly disclosed fallback; do
not describe blinking/head-turn checks as validated PAD.

**Pass:** UI reports PAD PASS/FAIL/UNAVAILABLE truthfully.

## Phase 6: attendance and persistence

Use SQLite. Implement named sessions, idempotent writes, database-level
uniqueness for accepted student/session pairs, event history, manual
review/audit events, session history and CSV export. Never let video
frames create duplicate records.

**Pass:** tests cover duplicates, unknowns, uncertain matches, PAD
rejection, manual review, restart persistence and session state.

## Phase 7: dashboard

Prefer Streamlit for speed unless existing code already uses a different
stack. Screens: dashboard, students/enrollment, live attendance, review
queue, session history, export and diagnostics. Synthetic fixtures must
be labelled and must never masquerade as live predictions.

## Phase 8: experiments and paper

Export anonymized predictions and ground truth. Compute metrics from raw
CSV using scripts. Compare recognition-only vs confidence review vs
recognition+PAD only when PAD actually runs. Capture screenshots from
the real app. Do not hardcode charts or results.

## Phase 9: final checks

Clean install; camera test; model smoke tests; unit tests;
enrollment→session→recognition→attendance→review/export end-to-end;
duplicate prevention; unknown face; photo/replay test if PAD exists;
restart persistence; CSV correctness.

Deliver source, pinned requirements, Windows setup README, model/license
manifest, tests report, anonymized raw results, verified literature
matrix, and paper draft consistent with code and data.
