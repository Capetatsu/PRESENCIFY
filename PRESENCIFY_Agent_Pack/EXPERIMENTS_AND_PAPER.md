# PRESENCIFY --- Evaluation Protocol and Paper Plan

## Research question

Does adding PAD and confidence-based review to a recent lightweight
face-recognition pipeline reduce incorrect automatic attendance
decisions compared with recognition-only, on a CPU webcam prototype
under controlled test conditions?

This is a hypothesis. If PAD is not implemented, revise the question
before final experiments.

## Hypotheses

-   H1: confidence review reduces incorrect automatic decisions but
    increases review load.
-   H2: a working PAD model reduces acceptance of tested photo/replay
    attacks.
-   H3: the selected model achieves measurable latency on the target
    CPU.
-   H4: error rates vary with lighting, pose and distance.

Unsupported hypotheses are valid results; do not manufacture success.

## Protocol

-   Use consenting participants and pseudonymous IDs such as P001.
-   Split by capture session/condition, not adjacent video frames.
-   Keep validation data separate for threshold selection and final test
    data.
-   Test frontal/moderate angles, normal/low indoor lighting, near/far
    distance, enrolled people, unknown people, no-face frames and
    multiple faces if supported.
-   For PAD, test genuine live faces, printed photos and screen replays
    separately.
-   Describe small data as a controlled pilot, not a generalisable
    classroom study.
-   Avoid storing unnecessary video. If retained, obtain consent,
    restrict access and document deletion.

## Comparisons

A. Recognition-only baseline. B. Recognition + confidence-based manual
review. C. Recognition + PAD + review, only if a real PAD model runs.

Use the same test cases across conditions. Report PAD-unavailable
separately; it is not a pass.

## Metrics

Recognition/attendance: precision, recall, F1, confusion matrix, false
acceptance/match count, false rejection/missed match count, unknown
handling, review rate, duplicate attempts blocked, session assignment
correctness.

PAD where protocol supports it: APCER, BPCER, ACER, EER/HTER as defined
by the paper/protocol. Name tested attack types. Passing photo tests
does not prove resistance to masks, deepfakes or unseen attacks.

Runtime: mean/median per-face latency, approximate end-to-end FPS, load
time, approximate RAM, CPU, package versions, input resolution and frame
skip.

## Thresholds

Similarity is not a probability. Select thresholds on validation data,
freeze them before final test, report threshold and selection method,
and keep preprocessing/model version fixed.

## Reproducibility files

-   `experiments/literature_matrix.csv`
-   `experiments/predictions_anonymized.csv`
-   `experiments/metrics.json`
-   `experiments/environment.txt`
-   `experiments/model_manifest.json`
-   `experiments/test_protocol.md`
-   `experiments/limitations.md`

Model manifest: paper, official repo, checkpoint filename/hash, license,
version, preprocessing, input size, device, model load status and
latency.

## Paper outline

1.  Title
2.  Abstract (write last)
3.  Keywords
4.  Introduction
5.  Related work (2024--2026 only for core cited work, per requirement)
6.  Narrow research gap supported by papers
7.  Research question and actual contributions
8.  Architecture
9.  Models and implementation
10. Experimental setup, machine, participant protocol, splits and
    thresholds
11. Results from actual logs
12. Discussion and comparable literature
13. Limitations
14. Ethics and privacy
15. Conclusion/future work
16. Verified references with DOI/official URLs

Never claim "100% accurate," "prevents all proxy attendance," "cannot be
spoofed," "works in every classroom," "first-ever system," or
"real-time" without measured evidence. Do not present published
benchmark numbers as PRESENCIFY results.
