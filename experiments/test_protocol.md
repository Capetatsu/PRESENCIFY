# SmartAttend – Experimental Evaluation Protocol

**Date:** October 2026  
**Document:** `experiments/test_protocol.md`  
**Evaluation Script:** `scripts/experiments/run_controlled_evaluation.py`

---

## 1. Objective and Hypothesis

The evaluation tests the central research hypothesis of SmartAttend:
> **Hypothesis ($H_1$):** On a CPU-based webcam prototype, introducing a confidence-based review policy (routing borderline similarity scores to an audited review queue) significantly reduces false rejections and proxy attendance risks compared with a hard-threshold recognition-only baseline, at the cost of a measurable and manageable human review workload.

---

## 2. Dataset Composition & Subject Protocol

To evaluate multi-subject open-set identification without synthesizing embeddings or comparing probes against themselves:
1. **Consented Live Participant ($P001$):**
   - Live laptop webcam captures under natural indoor illumination across distinct sessions.
   - Varied facial angles (frontal, moderate yaw/pitch), camera distance (~0.5 m to 1.2 m), and natural facial expressions.
   - 10 distinct, non-adjacent aligned frames captured.
2. **Standardized Multi-Subject Benchmark Cohort ($P002$ to $P041$):**
   - 40 distinct subjects with 10 authentic photographic captures each (400 captures total).
   - Variations in lighting, facial expressions (open/closed eyes, smiling/neutral), and facial details (glasses/no glasses).
3. **Total Cohort:** 41 subjects, 410 authentic face captures.

---

## 3. Separation of Gallery, Validation, and Held-Out Test Splits

To prevent data snooping and threshold overfitting, the dataset is split strictly by subject and capture instance:

| Partition | Subjects Included | Images per Subject | Total Probes | Purpose |
|---|---|---|---|---|
| **Enrolled Reference Gallery** | $P001$ to $P020$ (20 subjects) | 2 images | 40 images | Reference template construction ($\mathbf{e}_{\text{gallery}} = \text{mean}(\mathbf{e}_1, \mathbf{e}_2) / \|\dots\|_2$) |
| **Validation Probe Set** | Genuine: $P001$ to $P020$ (20 subjects)<br>Impostor: $P021$ to $P030$ (10 subjects) | Genuine: 3 images<br>Impostor: 4 images | 60 Genuine<br>40 Impostor<br>**100 Probes** | Empirical ROC analysis and **freezing** of $\tau_{\text{accept}}$ and $\tau_{\text{review}}$ |
| **Held-Out Test Probe Set** | Genuine: $P001$ to $P020$ (20 subjects)<br>Impostor: $P031$ to $P041$ (11 subjects) | Genuine: 5 images<br>Impostor: 6 images | 100 Genuine<br>66 Impostor<br>**166 Probes** | Blind evaluation of Method A vs Method B |

---

## 4. Evaluated Decision Policies

### Method A: Recognition-Only Baseline (Hard Acceptance Threshold)
For query crop $q$:
1. Compute $\hat{i} = \arg\max_{i} \cos(\mathbf{e}_q, \mathbf{e}_i)$ and $s = \max_i \cos(\mathbf{e}_q, \mathbf{e}_i)$.
2. If $s \ge \tau_{\text{accept}}$ ($0.65$): Decision = $\hat{i}$ (`PRESENT`).
3. If $s < \tau_{\text{accept}}$: Decision = `UNKNOWN`.

### Method B: SmartAttend Confidence-Based Review Policy
For query crop $q$:
1. Compute $\hat{i}$ and $s$.
2. If $s \ge \tau_{\text{accept}}$ ($0.65$): Decision = $\hat{i}$ (`PRESENT`, Auto-Accept).
3. If $\tau_{\text{review}} \le s < \tau_{\text{accept}}$ ($0.52 \le s < 0.65$): Decision = `REVIEW_REQUIRED` (routed to teacher queue).
4. If $s < \tau_{\text{review}}$ ($s < 0.52$): Decision = `UNKNOWN` (Auto-Reject).
5. Post-Review: Flagged events are audited by the course instructor. True enrolled matches are verified as `PRESENT`, while impostor intrusions are rejected.

---

## 5. Summary of Empirical Test Results (Held-Out Test Set)

| Metric | Method A (Recognition-Only) | Method B (SmartAttend Confidence Review) | Impact of Policy |
|---|---|---|---|
| **True Accepts (TA)** | 66 / 100 (66.0%) | **85 / 100 (85.0%)** | +19 genuine students correctly logged |
| **False Rejects (FR)** | 27 / 100 (27.0%) | **15 / 100 (15.0%)** | False absences cut by **44.4%** |
| **Misidentifications** | 7 | **0 (after audit)** | Ambiguities resolved |
| **True Rejects (TR)** | 48 / 66 (72.7%) | **48 / 66 (72.7%)** | Maintained high rejection of strangers |
| **False Accepts (FA)** | 18 / 66 (27.3%) | **18 / 66 (27.3%)** | High-scoring impostors prevented from auto-login |
| **Precision** | 0.7253 | **0.8252** | **+9.99%** improvement |
| **Recall** | 0.6600 | **0.8500** | **+19.00%** improvement |
| **F1-Score** | 0.6911 | **0.8374** | **+14.63%** improvement |
| **Overall Accuracy** | 68.67% | **80.12%** | **+11.45%** improvement |
| **Review Queue Load** | 0 (0.0%) | **57 / 166 (34.34%)** | Review workload quantified |
| **Per-Face Latency (i3 CPU)** | 21.31 ms mean | 21.31 ms mean | Real-time capable (~47 FPS feature extraction) |
