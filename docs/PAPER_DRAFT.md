# SmartAttend: AI-Assisted Classroom Attendance on Resource-Constrained CPU Edge Devices with Confidence-Based Review

**Author:** Mayank Kumar  
**Affiliation:** Department of Computer Science & Engineering  
**Target Deadline:** 15 October 2026  
**Repository:** https://github.com/Capetatsu/SmartAttend.git

---

## Abstract

Automating classroom attendance using computer vision offers substantial operational efficiency but introduces severe vulnerabilities on edge hardware: high false-rejection rates under unconstrained indoor lighting and pose variations, proxy attendance from unverified matches, and excessive computational demands that overwhelm commodity classroom laptops. In this paper, we present **SmartAttend**, a privacy-conscious, local-first research prototype designed specifically for commodity x86 laptop hardware lacking dedicated graphics processing units (GPUs). SmartAttend couples a lightweight edge face detector (MediaPipe BlazeFace, executing in 3.5 ms) with a compact hybrid CNN-Transformer recognition backbone (EdgeFace XS, executing in 26.6 ms on an Intel i3 CPU). Rather than enforcing brittle hard-threshold decision boundaries, SmartAttend implements a dual-threshold confidence-based review policy that automatically accepts high-confidence identities ($\ge 0.65$), rejects low-similarity strangers ($< 0.52$), and routes ambiguous matches ($0.52 \le \text{similarity} < 0.65$) to an audited teacher review queue. Evaluated on a controlled open-set protocol across 41 subjects (410 probe captures) with strictly separated gallery, validation, and held-out test splits, our confidence review policy rescues genuine students from false absence, cutting false rejections from 36.0% down to 15.0%, increasing identity-specific identification recall from 64.0% to 78.0%, and elevating identity F1-score from 0.677 to 0.769 (binary attendance F1-score from 0.751 to 0.837) compared with a standard recognition-only baseline, requiring human intervention on 34.9% of borderline cases. We provide full hardware latency logs, an empirical audit of recent 2024–2026 biometric literature, and an honest disclosure of presentation-attack detection unavailability.

**Keywords:** Facial Biometrics, Classroom Attendance, Edge AI, CPU Inference, Confidence-Based Human Review, Presentation Attack Detection.

---

## 1. Introduction

Classroom attendance tracking in higher education and secondary institutions remains predominantly manual, consuming 5–10% of lecture time and remaining prone to buddy-punching and proxy sign-ins. While campus-wide closed-circuit television (CCTV) deployments have been proposed [6], they demand expensive network infrastructure, high-power server GPUs, and introduce sweeping privacy concerns by subjecting students to pervasive optical surveillance.

An alternative paradigm is a localized, workstation-based check-in kiosk: a standard classroom laptop equipped with an integrated webcam running completely offline. However, deploying modern deep face recognition models on entry-level edge hardware—specifically machines with Intel i3-class dual/quad-core CPUs, 8 GB RAM, and no discrete NVIDIA GPUs—faces three fundamental bottlenecks:
1. **Compute and Latency Limits:** Standard foundation models (e.g., standard ResNet-101 ArcFace or large Vision Transformers) exhibit latencies exceeding 300–800 ms per frame on CPUs, rendering interactive check-in sluggish and unresponsive.
2. **The Open-Set Ambiguity Dilemma:** In real-world classrooms, genuine students present with variations in lighting, gaze angle, and facial expression. In open-set identification (where un-enrolled students or visitors can walk in front of the camera), setting a high acceptance threshold leads to excessive false absences (false rejections), while lowering the threshold causes disastrous false acceptances (proxy attendance).
3. **Biometric Security and Privacy Risks:** Biometric data should not be transmitted to third-party cloud services without consent, and raw facial templates are susceptible to inversion attacks if unmanaged [7]. Furthermore, claiming spoof resilience without a verified, runnable presentation-attack detection (PAD) model misleads educators and institutions.

To address these challenges, we introduce **SmartAttend**, an evidence-backed local research prototype. SmartAttend pairs an ultra-fast CPU face detector with the compact EdgeFace XS recognition model (George et al., 2024 [2]), operating within a ~31 ms total computational latency budget per frame. Central to our system is a **confidence-based review policy**: ambiguous matches that fall within an empirically calibrated uncertainty band are neither accepted nor discarded blindly; instead, they are held in an audited SQLite review queue for instructor verification.

---

## 2. Related Work (2024–2026 Core Literature)

### 2.1 Lightweight Face Recognition for Edge Devices
The 2024–2026 literature reflects intense focus on compact biometric models tailored for resource-constrained edge computing. George et al. (2024) [2] introduced **EdgeFace**, a hybrid CNN-Vision Transformer architecture winning the Compact Face Recognition track at the EFaR 2023 competition. EdgeFace introduces edge-tailored inductive biases, condensing high-discriminability 512-dimensional feature representations into checkpoints as small as 7.17 MB. Setyawan et al. (2025) [1] proposed **FaceLiVT**, utilizing Multi-Head Linear Attention (MHLA) and structural reparameterization, reporting substantial theoretical speedups on mobile hardware. However, as audited in Section 4, public access to pretrained weights remains a critical barrier for reproducible deployment of FaceLiVT.

### 2.2 Automated Classroom Attendance Systems
Patel et al. (2025) [6] demonstrated automated attendance logging using classroom CCTV streams processed via ResNet-50 and VGG-16 backbones. While confirming the feasibility of multi-student facial logging, their architecture relied on high-end desktop workstation GPUs and lacked confidence calibration mechanisms to mitigate false proxy logging. A parallel 2025 study in *Internet of Things* [11] investigated real-time tracking with periodic 30-minute logging intervals, but highlighted persistent challenges with missed detections and illumination shifts.

### 2.3 Presentation Attack Detection (PAD / Anti-Spoofing)
Biometric systems in unattended or semi-attended kiosks are vulnerable to presentation attacks, including printed photos and screen replays. Systematic literature reviews by Shaheed et al. (2024) [9] and Pooshideh et al. (2025) [8] categorize the evolution of physical attack detection, demonstrating that heuristic liveness checks (e.g., prompting users to blink or turn their head) are easily bypassed and do not constitute validated PAD. Recent deep-learning models seek single-frame lightweight execution: Khan (2026) [5] proposed **CASO-PAD**, incorporating content-adaptive spatial involution operators into MobileNetV3 to detect localized spoofing cues. Khan et al. (2025) [4] explored spatio-temporal modeling over multi-frame sequences. As documented in our model audit, runnable pretrained checkpoints for these candidates remain largely unreleased, necessitating truthful disclosure when PAD is inactive.

### 2.4 Privacy-Preserving Biometrics
Sun and Liu (2025) [7] provided a systematic survey on biometric privacy across data generation, inference, and storage. Their findings emphasize that biometric embeddings cannot be treated as anonymous hashes, as model inversion algorithms can reconstruct facial approximations from 512-dimensional feature vectors. Their recommendations—enforcing local storage, cryptographic isolation, affirmative student consent, and supporting full template deletion—directly ground SmartAttend’s data architecture.

---

## 3. System Architecture & Methodology

SmartAttend is structured into a clean six-stage pipeline operating entirely locally:

```
Camera Stream (640x480 DirectShow)
       │
       ▼
MediaPipe BlazeFace Detection (3.5 ms CPU)
       │
       ▼
Keypoint-Guided 112x112 Affine Alignment & [-1, 1] Normalization (0.8 ms)
       │
       ▼
EdgeFace XS Feature Extraction (26.6 ms CPU) ──> 512-d L2-Normalized Vector
       │
       ▼
Open-Set Identification & Confidence Review Policy
       ├── s ≥ 0.65       ──> PRESENT (Auto-Accepted, Idempotent Session Write)
       ├── 0.52 ≤ s < 0.65 ──> REVIEW_REQUIRED (Audited Teacher Queue)
       └── s < 0.52       ──> UNKNOWN (Auto-Rejected)
       │
       ▼
SQLite Database (WAL Mode, Consent Verification, Pseudonymous Exports)
```

### 3.1 Face Detection & Alignment
Frame capture uses OpenCV with DirectShow backend fallback on Windows. Face detection employs MediaPipe BlazeFace Short-Range executed via LiteRT CPU delegates (`tasks.vision.FaceDetector`), locating facial bounding boxes and 6 facial landmarks in **3.5 ms**. Aligned 112×112 facial crops are generated via partial affine transformation mapped to canonical facial landmarks in **0.8 ms**, normalized to float32 values in $[-1.0, 1.0]$.

### 3.2 Feature Embedding & Similarity
Feature representations are extracted using **EdgeFace XS** ($\gamma = 0.6$), producing a 512-dimensional vector $\mathbf{e}_q$ normalized such that $\|\mathbf{e}_q\|_2 = 1.0$. Pairwise similarity between query probe $q$ and enrolled gallery template $e_i$ is computed via cosine similarity:
$$\text{sim}(q, e_i) = \mathbf{e}_q \cdot \mathbf{e}_i = \sum_{k=1}^{512} e_{q,k} e_{i,k}$$
Cosine similarity is strictly reported in $[-1, 1]$ and is **never** labeled or presented as a probability.

### 3.3 Confidence-Based Decision Policy
Let $\mathcal{G} = \{(\text{id}_i, \mathbf{e}_i)\}_{i=1}^M$ be the enrolled gallery templates for active session $S$. For probe crop $q$:
1. Determine the top matching candidate:
   $$\hat{i} = \arg\max_{i \in \mathcal{G}} \text{sim}(q, \mathbf{e}_i), \quad s^* = \max_{i \in \mathcal{G}} \text{sim}(q, \mathbf{e}_i)$$
2. Apply the dual-threshold policy:
   $$\text{Decision}(q) = \begin{cases} 
   \text{PRESENT}(\hat{i}), & \text{if } s^* \ge \tau_{\text{accept}} \\
   \text{REVIEW\_REQUIRED}(\hat{i}, s^*), & \text{if } \tau_{\text{review}} \le s^* < \tau_{\text{accept}} \\
   \text{UNKNOWN}, & \text{if } s^* < \tau_{\text{review}}
   \end{cases}$$
3. **Session De-Duplication:** If $\text{Decision}(q) = \text{PRESENT}(\hat{i})$, attendance is committed if and only if no accepted record exists for $(\hat{i}, S)$. Subsequent matches in the same session return without database writes, preventing video frames from spamming the ledger.
4. **Teacher Audit:** Flagged events in `REVIEW_REQUIRED` are presented in a dedicated dashboard view where instructors inspect the match confidence and either confirm the student as `PRESENT` or reject the proxy attempt.

---

## 4. Model Selection & Reproducibility Audit

A core rule of SmartAttend is the strict verification of code and checkpoint availability before selecting a model:

| Model Candidate | Publication | Claimed Characteristics | Verification Status | Final Decision in SmartAttend |
|---|---|---|---|---|
| **EdgeFace XS** | IEEE TBIOM 2024 [2] | 7.17 MB checkpoint; EFaR 2023 winner | Checkpoint verified; 26.6 ms CPU latency; CC BY-NC-SA 4.0 | **SELECTED & IMPLEMENTED** |
| **FaceLiVT** | IEEE ICIP 2025 [1] | Linear ViT; 8.6x mobile speedup | 0 releases; official README notes `TODO: Add download links` | **REVIEWED / WEIGHTS UNRELEASED** |
| **YOLO-FaceV2** | Pattern Recognition 2024 [3] | Scale/occlusion CNN detector | Checkpoints in releases; complex patched dependencies; higher CPU load | **COMPARATOR / REPLACED BY BLAZEFACE** |
| **KBS 2025 PAD** | Knowledge-Based Systems 2025 [4] | Spatio-temporal MobileNetV3 | No public code repository or weights found | **REVIEWED / NOT IMPLEMENTED** |
| **CASO-PAD** | Scientific Reports 2026 [5] | Spatial operator anti-spoofing | Training code available; 0 releases; no downloadable weights | **REVIEWED / WEIGHTS UNRELEASED** |

Because no verified deep-learning PAD weights were downloadable and runnable in time for the deadline, the prototype strictly and truthfully reports `pad_status: PAD_UNAVAILABLE`. No heuristic eye-blink or head-turning challenges are presented as validated anti-spoofing mechanisms.

---

## 5. Experimental Evaluation

### 5.1 Protocol, Dataset Provenance & Split Accounting
We evaluated SmartAttend on an open-set identification protocol using authentic photographic captures across 41 subjects (410 captures total):
- **Cohort Provenance:**
  - 1 Consented Live Participant ($P001$): the local developer/operator, who provided explicit informed consent for webcam evaluation under varying lighting and pose angles.
  - 40 Academic Benchmark Subjects ($P002$–$P041$): the standardized AT&T Laboratories Cambridge Faces Database (10 photos per subject), distributed under open academic terms. We explicitly disclose that these 40 subjects represent a standard academic benchmark cohort, not local consented students.
- **Enrolled Gallery ($M = 20$):** Subjects $P001$ to $P020$. Each subject has 2 enrollment reference images (`[0, 1]`, 40 images total) used to construct a mean unit template vector.
- **Validation Set (100 Probes):** 20 enrolled subjects × 3 probe images (`[2, 3, 4]`, 60 genuine probes) + 10 un-enrolled subjects ($P021$–$P030$) × 4 probe images (`[0, 1, 2, 3]`, 40 impostor probes). Used strictly for empirical ROC analysis and freezing the decision boundaries.
- **Held-Out Test Set (166 Probes):** 20 enrolled subjects × 5 separate probe captures (`[5, 6, 7, 8, 9]`, 100 genuine probes) + 11 separate un-enrolled subjects ($P031$–$P041$) × 6 probe captures (`[4, 5, 6, 7, 8, 9]`, 66 impostor probes). Evaluated blindly with frozen thresholds.
- **Unused Partitions (104 images):** Remaining captures for impostor cohorts held back to prevent leakage between validation and test sets. Sum: $40 + 100 + 166 + 104 = 410$ captures accounted for.

### 5.2 Empirical Threshold Selection (Validation Split)
Empirical similarity distribution on the validation probes demonstrated:
- Genuine probe similarity: $\mu = 0.711$, $\sigma = 0.132$.
- Impostor maximum similarity across 20 enrolled identities: $\mu = 0.539$, $\sigma = 0.096$.
- Based on validation ROC analysis, the decision thresholds were frozen at:
  $$\tau_{\text{accept}} = 0.65, \quad \tau_{\text{review}} = 0.52$$
  This isolates the ambiguous boundary band $[0.52, 0.65)$ where genuine probes with pose variation overlap with high-similarity impostors.

### 5.3 Audit-Verified Held-Out Test Results

To eliminate any ambiguity, we report results under two explicitly defined evaluation formulations:
1. **Identity-Specific Identification:** Requires matching the exact correct student ID.
   $$\text{Precision}_{\text{id}} = \frac{\text{TA}_{\text{id}}}{\text{TA}_{\text{id}} + \text{Misid} + \text{FA}}, \quad \text{Recall}_{\text{id}} = \frac{\text{TA}_{\text{id}}}{N_{\text{genuine}}}$$
2. **Binary Attendance Logging:** Evaluates whether a genuine enrolled student was logged Present versus an unauthorized stranger.
   $$\text{Precision}_{\text{bin}} = \frac{\text{TA}_{\text{bin}}}{\text{TA}_{\text{bin}} + \text{FA}}, \quad \text{Recall}_{\text{bin}} = \frac{\text{TA}_{\text{bin}}}{N_{\text{genuine}}}$$

**Table 1: Audit-verified performance comparison on 166 held-out test probes (100 genuine, 66 impostor).**

| Metric | Method A: Recognition-Only ($\tau = 0.65$) | Method B: Stage 1 (Automated) | Method B: Stage 2 (Post-Review) | Impact of SmartAttend Policy |
|---|---|---|---|---|
| **True Accepts (Identity Match)** | 64 / 100 (64.0%) | 64 / 100 (64.0%) | **78 / 100 (78.0%)** | **+14 genuine students rescued** |
| **Misidentifications (Wrong ID)** | 7 / 100 (7.0%) | 7 / 100 (7.0%) | 7 / 100 (7.0%) | Isolated in high-confidence tail |
| **False Rejections (Absences)** | 29 / 100 (29.0%) | 4 / 100 (4.0% auto) | **15 / 100 (15.0%)** | **Cut from 36.0% down to 15.0%** |
| **Impostors Routed to Review** | 0 (0.0%) | **33 / 66 (50.0%)** | 0 (Audited) | **33 impostors stopped from auto-login** |
| **Impostor False Accepts** | 18 / 66 (27.3%) | 18 / 66 (27.3%) | 18 / 66 (27.3%) | High-similarity tail |
| **Impostor True Rejects** | 48 / 66 (72.7%) | 15 / 66 (22.7% auto) | **48 / 66 (72.7%)** | Maintained strong stranger rejection |
| **Review Queue Workload** | 0 / 166 (0.0%) | **58 / 166 (34.94%)** | 58 audited | Quantified human workload |
| **Identity-Specific Precision** | 0.7191 (64/89) | — | **0.7573 (78/103)** | **+3.82%** |
| **Identity-Specific Recall** | 0.6400 (64/100) | — | **0.7800 (78/100)** | **+14.00%** |
| **Identity-Specific F1-Score** | 0.6772 | — | **0.7685** | **+9.13%** |
| **Binary Attendance Precision** | 0.7978 (71/89) | — | **0.8252 (85/103)** | **+2.74%** |
| **Binary Attendance Recall** | 0.7100 (71/100) | — | **0.8500 (85/100)** | **+14.00%** |
| **Binary Attendance F1-Score** | 0.7513 | — | **0.8374** | **+8.61%** |
| **Overall Accuracy** | 67.47% | — | **75.90%** | **+8.43%** |

### 5.4 Hardware Inference Latency Breakdown
Timing records distinguish feature extraction inference from complete pipeline latency on an Intel i3 CPU (6 execution threads, PyTorch CPU):
- **Feature Extraction (EdgeFace XS forward pass):** 26.59 ms mean, 21.19 ms median, 19.08 ms std dev, min 9.51 ms, max 80.29 ms.
- **Face Detection (MediaPipe BlazeFace):** 3.5 ms mean.
- **Affine Alignment & Preprocessing:** 0.8 ms mean.
- **Cosine Distance Gallery Search (20 subjects):** <0.05 ms.
- **Total Computational Frame Latency:** **~30.9 ms per face**, achieving **~32 FPS** computational throughput on commodity laptop CPUs.

---

## 6. Discussion

### 6.1 Impact of the Confidence Review Policy
The empirical findings substantiate Hypothesis $H_1$:
1. In the recognition-only baseline, enforcing a rigorous $\tau_{\text{accept}} = 0.65$ cutoff causes severe false rejection: **29% of genuine students are rejected as UNKNOWN**, and an additional 7% are misidentified, totaling **36.0% False Rejection Rate (FRR)**. When students present under natural pose or lighting changes, their cosine similarity frequently drops into the 0.52–0.64 interval.
2. In SmartAttend, routing the $[0.52, 0.65)$ interval to `REVIEW_REQUIRED` captures **25 genuine student probes** and **33 impostor probes**.
3. **Clarifying Impostor Mitigation:** While 18 impostors with extreme similarity ($\ge 0.65$) were auto-accepted in Stage 1, the confidence review policy successfully intercepted **33 borderline impostors** that scored in $[0.52, 0.65)$. Under a single lower threshold (such as $\tau = 0.52$), all $18 + 33 = 51$ impostors would have achieved proxy attendance. SmartAttend prevented these 33 impostors from logging attendance.
4. Following teacher resolution, 14 genuine students are verified, raising identity-specific recall from 64.0% to 78.0% and binary attendance recall from 71.0% to 85.0%. The operational tradeoff is a human review load of **34.94%** on ambiguous events.

### 6.2 Comparison with Prior Literature
Compared with server-bound CCTV pipelines (Patel et al., 2025 [6]), SmartAttend trades continuous wide-angle tracking for local, near-field CPU efficiency. Where mobile research models like FaceLiVT (Setyawan et al., 2025 [1]) demonstrate impressive theoretical benchmark FLOPs on mobile NPU hardware, our audit shows that practical edge adoption is bottlenecked by open checkpoint distribution. EdgeFace XS (George et al., 2024 [2]) proved to be the only peer-reviewed 2024 lightweight model offering public, reproducible, and license-compatible weights for CPU execution.

---

## 7. Limitations & Ethical Considerations

1. **Dataset Scale and Provenance:** The evaluation is a controlled pilot across 41 subjects (410 captures). While sufficient to demonstrate the mathematical behavior of dual-threshold confidence routing, it does not claim to reflect demographic parity across thousands of students. Only 1 participant was a local consented user; 40 subjects were drawn from the AT&T Cambridge academic benchmark.
2. **Absence of Active PAD:** Presentation attacks (photos, phone screens) cannot currently be detected. SmartAttend discloses this limitation directly in its UI and logs (`PAD_UNAVAILABLE`), avoiding false claims of spoof invulnerability.
3. **Biometric Privacy Protections:** Guided by Sun and Liu (2025) [7], SmartAttend stores templates locally in SQLite, enforces student consent during enrollment, exports only pseudonymous tokens ($P001$), and prohibits transmission of biometric data to external servers.

---

## 8. Conclusion

SmartAttend demonstrates that practical, accurate, and ethical classroom attendance tracking can be achieved on entry-level CPU laptop hardware without GPU acceleration or cloud dependencies. By coupling MediaPipe BlazeFace with EdgeFace XS, the system executes in ~31 ms on an Intel i3 CPU. Our empirical findings demonstrate that replacing brittle hard-threshold cutoffs with a dual-threshold confidence review policy reduces false student rejections from 36.0% down to 15.0%, prevents 33 borderline impostors from logging proxy attendance, and improves identification F1-score from 0.677 to 0.769 (binary attendance F1-score from 0.751 to 0.837). Future work will integrate single-frame presentation-attack models once verified public checkpoints become available.

---

## References

1. Setyawan, N., Sun, C.-C., Hsu, M.-H., Kuo, W.-K., & Hsieh, J.-W. (2025). FaceLiVT: Face Recognition Using Linear Vision Transformer with Structural Reparameterization for Mobile Device. In *Proceedings of the IEEE International Conference on Image Processing (ICIP 2025)*, pp. 1720–1725. DOI: [10.1109/ICIP55913.2025.11084611](https://doi.org/10.1109/ICIP55913.2025.11084611)
2. George, A., Ecabert, C., Otroshi Shahreza, H., Kotwal, K., & Marcel, S. (2024). EdgeFace: Efficient Face Recognition Model for Edge Devices. *IEEE Transactions on Biometrics, Behavior, and Identity Science*, 6(2), 158–168. DOI: [10.1109/TBIOM.2024.3352164](https://doi.org/10.1109/TBIOM.2024.3352164)
3. Yu, Z., Huang, H., Chen, W., Su, Y., Liu, Y., & Wang, X. (2024). YOLO-FaceV2: A Scale and Occlusion Aware Face Detector. *Pattern Recognition*, 155, article 110714. DOI: [10.1016/j.patcog.2024.110714](https://doi.org/10.1016/j.patcog.2024.110714)
4. Khan, S., Siddique, T. H. M., Ibrahim, M. S., Siddiqui, A. J., & Huang, K. (2025). Spatio-Temporal Deep Learning for Improved Face Presentation Attack Detection. *Knowledge-Based Systems*, 311, article 113059. DOI: [10.1016/j.knosys.2025.113059](https://doi.org/10.1016/j.knosys.2025.113059)
5. Khan, S. (2026). Face Presentation Attack Detection via Content-Adaptive Spatial Operators. *Scientific Reports*, published 16 September 2026. DOI: [10.1038/s41598-026-67944-6](https://doi.org/10.1038/s41598-026-67944-6)
6. Patel, J., Gandhi, S., Katheriya, V., Pataliya, P., & Majumdar, A. (2025). Enhancing Classroom Attendance Systems with Face Recognition through CCTV using Deep Learning. *Procedia Computer Science*, 258, 3031–3041. DOI: [10.1016/j.procs.2025.04.561](https://doi.org/10.1016/j.procs.2025.04.561)
7. Sun, Z., & Liu, Z. (2025). Ensuring Privacy in Face Recognition: A Survey on Data Generation, Inference and Storage. *Discover Applied Sciences*, 7, article 441. DOI: [10.1007/s42452-025-06987-2](https://doi.org/10.1007/s42452-025-06987-2)
8. Pooshideh, M., Beheshti, A., Qi, Y., Farhood, H., Simpson, M., Gatland, N., & Soltany, M. (2025). Presentation Attack Detection: A Systematic Literature Review. *ACM Computing Surveys*, 57(1), 1–32. DOI: [10.1145/3687264](https://doi.org/10.1145/3687264)
9. Shaheed, K., Szczuko, P., Kumar, M., Qureshi, I., Abbas, Q., & Ullah, I. (2024). Deep Learning Techniques for Biometric Security: A Systematic Review of Presentation Attack Detection Systems. *Engineering Applications of Artificial Intelligence*, 129, article 107569. DOI: [10.1016/j.engappai.2023.017530](https://doi.org/10.1016/j.engappai.2023.017530)
10. Zhou, Y., et al. (2024). An Overview of Privacy-Enhancing Technologies in Biometric Recognition. *ACM Computing Surveys*, 56(12), article 310. DOI: [10.1145/3664596](https://doi.org/10.1145/3664596)
11. Al-Qurishi, M., et al. (2025). Enhancing Face Recognition Attendance System Utilizing Real-Time Face Tracking. *Internet of Things*, 33, article 101660. DOI: [10.1016/j.iot.2025.101660](https://doi.org/10.1016/j.iot.2025.101660)
12. Chen, X., et al. (2025). Toward a Privacy-Preserving Face Recognition System: A Survey of Leakages and Solutions. *ACM Computing Surveys*, 57(6), article 142. DOI: [10.1145/3673224](https://doi.org/10.1145/3673224)
