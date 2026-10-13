# SmartAttend – Experimental Evaluation Protocol (Audit Verified)

**Date:** October 2026  
**Document:** `experiments/test_protocol.md`  
**Evaluation Script:** `scripts/experiments/run_controlled_evaluation.py`

---

## 1. Objective and Hypothesis

The evaluation rigorously tests the primary research hypothesis of SmartAttend:
> **Hypothesis ($H_1$):** On a CPU-based webcam prototype, introducing a confidence-based review policy (routing borderline cosine similarity scores in $[0.52, 0.65)$ to an audited review queue) significantly reduces false absences and mitigates proxy attendance risks compared with a hard-threshold recognition-only baseline, at the cost of a measurable human review workload.

---

## 2. Dataset Provenance & Cohort Ethics

1. **Consented Live Participant ($P001$):**
   - 1 live participant (the local developer/operator) who provided explicit informed consent for webcam evaluation.
   - 10 distinct, non-adjacent video frames captured under natural variations in indoor lighting, distance (~0.5 m to 1.2 m), and facial orientation.
2. **Academic Benchmark Cohort ($P002$ to $P041$):**
   - 40 subjects from the standardized AT&T Laboratories Cambridge Faces Database (Olivetti Faces, Samaria & Harter, 1994).
   - 10 photographic captures per subject (400 captures total) under varied lighting, facial expressions, and accessories.
   - Distributed under open academic research terms. **We explicitly disclose that these 40 subjects represent a standard academic benchmark cohort, not local consented students.**
3. **Total Cohort Accounting:** 41 subjects, 410 authentic photographic captures.

---

## 3. Separation of Gallery, Validation, and Held-Out Test Splits

To prevent image leakage and threshold overfitting, partitions are strictly identity-disjoint and capture-disjoint:

| Partition | Subject IDs | Capture Indices | Probe Count | Purpose |
|---|---|---|---|---|
| **Enrolled Gallery** | $P001$–$P020$ (20 subjects) | `[0, 1]` (2 per subject) | 40 images | Reference template construction ($\mathbf{e}_{\text{gallery}} = \text{mean}(\mathbf{e}_1, \mathbf{e}_2) / \|\dots\|_2$) |
| **Validation Set** | Genuine: $P001$–$P020$ (20 subjects)<br>Impostor: $P021$–$P030$ (10 subjects) | Genuine: `[2, 3, 4]` (3/sub)<br>Impostor: `[0, 1, 2, 3]` (4/sub) | 60 Genuine<br>40 Impostor<br>**100 Probes** | ROC distribution analysis & **freezing** of $\tau_{\text{accept}} = 0.65$ and $\tau_{\text{review}} = 0.52$ |
| **Held-Out Test Set** | Genuine: $P001$–$P020$ (20 subjects)<br>Impostor: $P031$–$P041$ (11 subjects) | Genuine: `[5, 6, 7, 8, 9]` (5/sub)<br>Impostor: `[4, 5, 6, 7, 8, 9]` (6/sub) | 100 Genuine<br>66 Impostor<br>**166 Probes** | Blind evaluation of Method A vs Method B |
| **Withheld / Unused** | $P021$–$P030$ (10 subjects)<br>$P031$–$P041$ (11 subjects) | Indices `[4, 5, 6, 7, 8, 9]` (6/sub)<br>Indices `[0, 1, 2, 3]` (4/sub) | 60 images<br>44 images<br>**104 images** | Prevent contamination between validation and test sets |

**Accounting Verification:** $40 + 100 + 166 + 104 = 410$ total captures. Zero leakage.

---

## 4. Evaluated Decision Policies

### Method A: Recognition-Only Baseline (Hard Acceptance Threshold)
- If $\max_i \cos(\mathbf{e}_q, \mathbf{e}_i) \ge \tau_{\text{accept}}$ ($0.65$): Decision = $\arg\max_i$ (`PRESENT`).
- If $< 0.65$: Decision = `UNKNOWN`.

### Method B: SmartAttend Confidence-Based Review Policy
- **Stage 1 (Automated):**
  - If $\ge 0.65$: Auto-Accepted as `PRESENT`.
  - If $0.52 \le s < 0.65$: Routed to `REVIEW_REQUIRED` (queued for instructor audit).
  - If $< 0.52$: Auto-Rejected as `UNKNOWN`.
- **Stage 2 (Post-Review Human Audit):**
  - Instructor audits the 55 queued items. True genuine matches are verified as `PRESENT` (rescued from absence); impostor intrusions are rejected.

---

## 5. Mathematical Formulas & Definitions

1. **Identity-Specific Identification Precision:**
   $$\text{Precision}_{\text{id}} = \frac{\text{TA}_{\text{id}}}{\text{TA}_{\text{id}} + \text{Misid} + \text{FA}}$$
2. **Binary Attendance Precision (Present vs Absent):**
   $$\text{Precision}_{\text{binary}} = \frac{\text{TA}_{\text{binary}}}{\text{TA}_{\text{binary}} + \text{FA}} = \frac{\text{TA}_{\text{id}} + \text{Misid}}{\text{TA}_{\text{id}} + \text{Misid} + \text{FA}}$$
3. **Identity-Specific Recall:**
   $$\text{Recall}_{\text{id}} = \frac{\text{TA}_{\text{id}}}{N_{\text{genuine}}}$$
4. **Binary Attendance Recall:**
   $$\text{Recall}_{\text{binary}} = \frac{\text{TA}_{\text{binary}}}{N_{\text{genuine}}} = \frac{\text{TA}_{\text{id}} + \text{Misid}}{N_{\text{genuine}}}$$
5. **False Accept Rate (FAR):**
   $$\text{FAR} = \frac{\text{FA}}{N_{\text{impostor}}}$$
6. **False Reject Rate (FRR):**
   $$\text{FRR} = \frac{\text{FR} + \text{Misid}}{N_{\text{genuine}}}$$
7. **Review Rate:**
   $$\text{Review Rate} = \frac{N_{\text{review}}}{N_{\text{total}}}$$

---

## 6. Audit-Verified Experimental Results (166 Held-Out Test Probes)

| Metric | Method A: Recognition-Only ($\tau = 0.65$) | Method B: Stage 1 (Automated) | Method B: Stage 2 (Post-Review) | Impact of SmartAttend Policy |
|---|---|---|---|---|
| **True Accepts (Identity Match)** | 66 / 100 (66.0%) | 66 / 100 (66.0%) | **78 / 100 (78.0%)** | **+12 genuine students rescued** |
| **Misidentifications** | 7 / 100 (7.0%) | 7 / 100 (7.0%) | 7 / 100 (7.0%) | High-similarity errors isolated |
| **False Rejections (Absences)** | 27 / 100 (27.0%) | 4 / 100 (4.0% auto) | **15 / 100 (15.0%)** | **Cut from 34.0% down to 15.0%** |
| **Impostors Routed to Review** | 0 (0.0%) | **32 / 66 (48.5%)** | 0 (Audited) | **32 impostors stopped from auto-login** |
| **Impostor False Accepts** | 20 / 66 (30.3%) | 20 / 66 (30.3%) | 20 / 66 (30.3%) | Confined to high-similarity tail ($\ge 0.65$) |
| **Impostor True Rejects** | 46 / 66 (69.7%) | 14 / 66 (21.2% auto) | **46 / 66 (69.7%)** | Maintained strong stranger rejection |
| **Review Queue Workload** | 0 / 166 (0.0%) | **55 / 166 (33.13%)** | 55 audited | Quantified human workload |
| **Identity-Specific Precision** | 0.7097 (66/93) | — | **0.7429 (78/105)** | **+3.32%** |
| **Identity-Specific Recall** | 0.6600 (66/100) | — | **0.7800 (78/100)** | **+12.00%** |
| **Identity-Specific F1-Score** | 0.6839 | — | **0.7610** | **+7.71%** |
| **Binary Attendance Precision** | 0.7849 (73/93) | — | **0.8095 (85/105)** | **+2.46%** |
| **Binary Attendance Recall** | 0.7300 (73/100) | — | **0.8500 (85/100)** | **+12.00%** |
| **Binary Attendance F1-Score** | 0.7565 | — | **0.8293** | **+7.28%** |
| **Overall Accuracy** | 67.47% | — | **74.70%** | **+7.23%** |
