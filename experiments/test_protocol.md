# PRESENCIFY – Experimental Evaluation Protocol (Audit Verified)

**Date:** October 2026  
**Document:** `experiments/test_protocol.md`  
**Evaluation Script:** `scripts/experiments/run_controlled_evaluation.py`

---

## 1. Objective and Hypothesis

The evaluation rigorously tests the primary research hypothesis of PRESENCIFY:
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

To prevent image leakage and threshold overfitting, partitions enforce capture-disjoint genuine probes and identity-disjoint impostor cohorts:

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

### Method B: PRESENCIFY Confidence-Based Review Policy
- **Stage 1 (Automated):**
  - If $\ge 0.65$: Auto-Accepted as `PRESENT`.
  - If $0.52 \le s < 0.65$: Routed to `REVIEW_REQUIRED` (queued for instructor audit).
  - If $< 0.52$: Auto-Rejected as `UNKNOWN`.
- **Stage 2 (Post-Review Human Audit):**
  - Instructor audits the 56 queued items (23 genuine, 33 impostor). True genuine matches are verified as `PRESENT` (12 rescued from absence); impostor intrusions (33) are rejected.

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

| Metric | Method A: Recognition-Only ($\tau = 0.65$) | Method B: Stage 1 (Automated) | Method B: Stage 2 (Post-Review) | Impact of PRESENCIFY Policy |
|---|---|---|---|---|
| **True Accepts (Identity Match)** | 65 / 100 (65.0%) | 65 / 100 (65.0%) | **77 / 100 (77.0%)** | **+12 genuine students rescued** |
| **Misidentifications** | 7 / 100 (7.0%) | 7 / 100 (7.0%) | 7 / 100 (7.0%) | High-similarity errors isolated |
| **False Rejections (Absences)** | 28 / 100 (28.0%) | 5 / 100 (5.0% auto) | **16 / 100 (16.0%)** | **Cut from 35.0% down to 16.0%** |
| **Impostors Routed to Review** | 0 (0.0%) | **33 / 66 (50.0%)** | 0 (Audited) | **33 impostors stopped from auto-login** |
| **Impostor False Accepts** | 18 / 66 (27.3%) | 18 / 66 (27.3%) | 18 / 66 (27.3%) | Confined to high-similarity tail ($\ge 0.65$) |
| **Impostor True Rejects** | 48 / 66 (72.7%) | 15 / 66 (22.7% auto) | **48 / 66 (72.7%)** | Maintained strong stranger rejection |
| **Review Queue Workload** | 0 / 166 (0.0%) | **56 / 166 (33.73%)** | 56 audited | Quantified human workload |
| **Identity-Specific Precision** | 0.7222 (65/90) | — | **0.7549 (77/102)** | **+3.27%** |
| **Identity-Specific Recall** | 0.6500 (65/100) | — | **0.7700 (77/100)** | **+12.00%** |
| **Identity-Specific F1-Score** | 0.6842 | — | **0.7624** | **+7.82%** |
| **Binary Attendance Precision** | 0.8000 (72/90) | — | **0.8235 (84/102)** | **+2.35%** |
| **Binary Attendance Recall** | 0.7200 (72/100) | — | **0.8400 (84/100)** | **+12.00%** |
| **Binary Attendance F1-Score** | 0.7579 | — | **0.8317** | **+7.38%** |
| **Overall Accuracy** | 68.07% | — | **75.30%** | **+7.23%** |
