# PRESENCIFY Literature Review --- 2024--2026

**Purpose:** shortlist of recent work to read for PRESENCIFY. Summaries
below are screening notes based on accessible publisher
metadata/abstracts and official repositories; they are not a claim that
every full text has been read cover-to-cover. Before citing detailed
methods or results, read the full paper and fill in the literature
matrix. All selected core references must fall within 2024--2026.

## Priority A --- Recognition, detection and classroom attendance

### 1. FaceLiVT: Face Recognition Using Linear Vision Transformer with Structural Reparameterization for Mobile Device (2025)

-   Authors: Novendra Setyawan, Chi-Chia Sun, Mao-Hsiu Hsu, Wen-Kai Kuo,
    Jun-Wei Hsieh.
-   Venue: IEEE ICIP 2025, pp. 1720--1725.
-   DOI: https://doi.org/10.1109/ICIP55913.2025.11084611
-   Official repo: https://github.com/novendrastywn/FaceLiVT
-   Relevance: hybrid CNN/Transformer for mobile/edge face recognition;
    benchmarks include LFW, CFP-FP, AgeDB-30, IJB-B and IJB-C.
-   Feasibility: official setup describes CUDA/DALI-oriented
    dependencies; inference example uses `map_location='cpu'`, but
    usable weights and CPU-compatible dependencies must be verified.
    FaceLiVTv2 is listed as under review in the repo; do not call it
    peer-reviewed without checking its current status.
-   Read: alignment, normalization, checkpoint availability/license,
    benchmarks and speed-measurement hardware.

### 2. EdgeFace: Efficient Face Recognition Model for Edge Devices (2024)

-   Authors: Anjith George, Christophe Ecabert, Hatef Otroshi Shahreza,
    Ketan Kotwal, Sébastien Marcel.
-   Venue: IEEE Transactions on Biometrics, Behavior, and Identity
    Science, 6(2), 158--168 (2024).
-   DOI: https://doi.org/10.1109/TBIOM.2024.3352164
-   Official repo: https://github.com/otroshi/edgeface
-   Relevance: lightweight hybrid CNN/Transformer face-recognition model
    with public pretrained variants.
-   Feasibility: practical comparator/fallback; still verify the
    selected variant, checkpoint, license, alignment and CPU speed.
    Publication is 2024 even though a preprint existed earlier.

### 3. YOLO-FaceV2: A Scale and Occlusion Aware Face Detector (2024)

-   Authors: Ziping Yu, Hongbo Huang, Weijun Chen, Yongxin Su, Yahui
    Liu, Xiuying Wang.
-   Venue: Pattern Recognition, 155, article 110714.
-   DOI: https://doi.org/10.1016/j.patcog.2024.110714
-   Publisher:
    https://www.sciencedirect.com/science/article/pii/S0031320324004655
-   Relevance: face detection under scale variation and occlusion.
-   Read: model size, WIDER FACE protocol, inference latency,
    code/weights availability.

### 4. A Lightweight Face Detector via Bi-Stream Convolutional Neural Network and Vision Transformer (2024)

-   Venue: Information, 15(5), 290.
-   Publisher: https://www.mdpi.com/2078-2489/15/5/290
-   Relevance: lightweight face detection and benchmark comparisons on
    WIDER FACE.
-   Read: parameters, hard/medium/easy results, CPU/edge deployment
    cost.

### 5. Multitask Learning Strategy with Pseudo-Labeling: Face Recognition, Facial Landmark Detection, and Head Pose Estimation (2024)

-   Venue: Sensors, 24(10), 3212.
-   Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11125148/
-   Relevance: recognition plus landmarks/head pose; useful for
    alignment and pose-related failure analysis.
-   Feasibility: reported training used an RTX A6000 for 60--120 hours;
    do not reproduce training on the laptop.

### 6. Enhancing Classroom Attendance Systems with Face Recognition through CCTV using Deep Learning (2025)

-   Authors: Jaykumar Patel, Savita Gandhi, Vishal Katheriya, Parth
    Pataliya, Ankan Majumdar.
-   Venue: Procedia Computer Science, 258, 3031--3041.
-   DOI: https://doi.org/10.1016/j.procs.2025.04.561
-   Publisher:
    https://www.sciencedirect.com/science/article/pii/S1877050925016655
-   Relevance: directly overlaps with CCTV classroom attendance; studies
    ResNet-50/VGG-16 with PCA feature extraction.
-   Implication: attendance via CCTV is not itself novel. Read data
    collection, split, camera geometry, errors and limitations.

### 7. Enhancing Face Recognition Attendance System Utilizing Real-Time Face Tracking (2025)

-   Venue: Internet of Things, 33, article 101660.
-   DOI: https://doi.org/10.1016/j.iot.2025.101660
-   Publisher:
    https://www.sciencedirect.com/science/article/pii/S254266052500174X
-   Relevance: real-time tracking and periodic attendance logic; paper
    describes logging at 30-minute intervals and a minimum-presence
    policy.
-   Read: how presence is defined, missed detections, false attendance
    and session logic.

### 8. An Integrated Facial Recognition System for Classroom Resource Optimization Using MobileNet and SSA-SVM (online 2025; issue 2026)

-   Venue: Scientific Reports, volume 16, article 265 (2026 issue);
    online 29 November 2025.
-   Publisher: https://www.nature.com/articles/s41598-025-29327-1
-   Relevance: classroom resource/attendance use with MobileNet feature
    extraction and SSA-SVM.
-   Date note: record online-first and issue dates accurately in the
    required citation style.
-   Read: what is novel, dataset, runtime and reproducibility.

## Priority B --- Presentation-attack detection (PAD / anti-spoofing)

### 9. Deep Learning Techniques for Biometric Security: A Systematic Review of Presentation Attack Detection Systems (2024)

-   Authors: Kashif Shaheed, Piotr Szczuko, Munish Kumar, Imran Qureshi,
    Qaisar Abbas, Ihsan Ullah.
-   Venue: Engineering Applications of Artificial Intelligence, 129,
    article 107569.
-   DOI: https://doi.org/10.1016/j.engappai.2023.107569
-   Publisher:
    https://www.sciencedirect.com/science/article/pii/S0952197623017530
-   Relevance: systematic PAD review across biometric modalities;
    extract face-specific evidence only.

### 10. Unmasking Deception: A Comprehensive Survey on the Evolution of Face Anti-Spoofing Methods (online 2024; issue 2025)

-   Venue: Neurocomputing, 617, article 128992 (2025 issue).
-   DOI: https://doi.org/10.1016/j.neucom.2024.128992
-   Publisher:
    https://www.sciencedirect.com/science/article/pii/S0925231224017636
-   Relevance: face anti-spoofing taxonomy, evolution and generalisation
    concerns.
-   Date note: publisher issue date is 7 February 2025; DOI contains
    2024.

### 11. Spatio-Temporal Deep Learning for Improved Face Presentation Attack Detection (2025)

-   Authors: Shujaat Khan, Taha Hasan Masood Siddique, Muhammad Sohail
    Ibrahim, Abdul Jabbar Siddiqui, Kejie Huang.
-   Venue: Knowledge-Based Systems, 311, article 113059.
-   DOI: https://doi.org/10.1016/j.knosys.2025.113059
-   Publisher:
    https://www.sciencedirect.com/science/article/pii/S0950705125001066
-   Relevance: lightweight MobileNetV3-based spatial/temporal method;
    evaluates Replay-Attack, Replay-Mobile and ROSE-Youtu.
-   Feasibility: find official code/weights and verify window length,
    preprocessing and CPU performance.
-   Read: cross-dataset generalisation, protocol, EER, model size and
    attack coverage.

### 12. Presentation Attack Detection: A Systematic Literature Review (2025)

-   Authors: Matineh Pooshideh, Amin Beheshti, Yuankai Qi, Helia
    Farhood, Mike Simpson, Nick Gatland, Mehdi Soltany.
-   Venue: ACM Computing Surveys, 57(1), pp. 1--32.
-   DOI: https://doi.org/10.1145/3687264
-   Institutional record:
    https://researchers.mq.edu.au/en/publications/presentation-attack-detection-a-systematic-literature-review/
-   Relevance: methods, datasets and protocols across PAD; useful for
    avoiding invalid cross-dataset comparisons.

### 13. A Survey of Deep Learning for Face Presentation Attack Detection (2025)

-   Venue: Neurocomputing, 652, article 131136.
-   DOI: https://doi.org/10.1016/j.neucom.2025.131136
-   Publisher:
    https://www.sciencedirect.com/science/article/pii/S0925231225018089
-   Relevance: spatial/frequency approaches, self-supervised methods,
    unknown attacks and generalisation.

### 14. Face Presentation Attack Detection via Content-Adaptive Spatial Operators (2026)

-   Author: Shujaat Khan.
-   Venue: Scientific Reports, published 16 September 2026.
-   DOI: https://doi.org/10.1038/s41598-026-67944-6
-   Full text: https://www.nature.com/articles/s41598-026-67944-6
-   Relevance: CASO-PAD, lightweight RGB single-frame MobileNetV3 with
    content-adaptive spatial operators.
-   Feasibility: verify official code/checkpoint. The paper's benchmark
    numbers are not PRESENCIFY results.
-   Read: dataset protocols, HTER/EER, model size/FLOPs and
    unknown-attack generalisation.

### 15. Unified Physical--Digital Face Attack Detection Challenge: A Review (2026)

-   Venue: IET Biometrics; first published 4 February 2026.
-   DOI: https://doi.org/10.1049/bme2/9653627
-   Publisher:
    https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/bme2/9653627
-   Relevance: discusses UniAttackData and distinguishes physical
    presentation attacks from digital attacks.
-   Read: which attacks are in-scope for a webcam demo and which remain
    future work.

### 16. Moving Masks: A Preliminary Study on Face Presentation Attack On-The-Move (WACV Workshops 2026)

-   Authors: Raghavendra Ramachandra et al.
-   Venue: IEEE/CVF WACV Workshops 2026, pp. 1635--1645.
-   Full text:
    https://openaccess.thecvf.com/content/WACV2026W/GAPBio/html/Ramachandra_Moving_Masks_A_Preliminary_Study_on_Face_Presentation_Attack_On-The-Move_WACVW_2026_paper.html
-   Relevance: evaluates moving-mask attacks with multiple cameras.
-   Limitation: not directly reproducible on a laptop webcam; use it to
    discuss threat-model limits.

## Priority C --- Privacy and responsible biometric handling

### 17. An Overview of Privacy-Enhancing Technologies in Biometric Recognition (2024)

-   Venue: ACM Computing Surveys, 56(12), article 310.
-   DOI: https://doi.org/10.1145/3664596
-   Relevance: framework for privacy-enhancing technologies in
    biometrics. Use it to justify local storage, access limits and
    deletion.

### 18. Ensuring Privacy in Face Recognition: A Survey on Data Generation, Inference and Storage (2025)

-   Authors: Zhifang Sun, Zhe Liu.
-   Venue: Discover Applied Sciences, 7, article 441.
-   DOI: https://doi.org/10.1007/s42452-025-06987-2
-   Full text:
    https://link.springer.com/article/10.1007/s42452-025-06987-2
-   Relevance: privacy risks and mitigations across data, inference and
    storage, including template leakage.

### 19. Toward a Privacy-Preserving Face Recognition System: A Survey of Leakages and Solutions (2025)

-   Venue: ACM Computing Surveys, 57(6).
-   DOI: https://doi.org/10.1145/3673224
-   Publisher: https://doi.org/10.1145/3673224
-   Relevance: privacy-preserving face recognition and leakage surfaces.
    Embeddings should not be described as anonymous.

### 20. A Review of Privacy-Preserving Biometric Identification and Authentication Protocols (2025)

-   Venue: Computers & Security, 150, article 104309.
-   DOI: https://doi.org/10.1016/j.cose.2024.104309
-   Publisher:
    https://www.sciencedirect.com/science/article/abs/pii/S0167404824006151
-   Relevance: protocols including homomorphic encryption and secure
    multi-party computation.
-   Scope note: advanced cryptography is not required for the short
    prototype; discuss as future work only.

### 21. Facial Privacy in the Digital Era: A Comprehensive Survey on Methods, Evaluation, and Future Directions (2025)

-   Venue: Computer Science Review, 58, article 100785.
-   DOI: https://doi.org/10.1016/j.cosrev.2025.100785
-   Publisher:
    https://www.sciencedirect.com/science/article/pii/S1574013725000619
-   Relevance: anonymisation, de-identification and privacy-preserving
    face methods.

### 22. Federated Learning for Biometric Recognition: A Survey (2024)

-   Venue: Artificial Intelligence Review, 57, article 208.
-   DOI: https://doi.org/10.1007/s10462-024-10847-7
-   Publisher:
    https://link.springer.com/article/10.1007/s10462-024-10847-7
-   Relevance: privacy-aware learning and decentralised biometric data.
-   Scope note: federated learning is not required in this local
    prototype.

## Synthesis: what the literature does and does not support

1.  CCTV classroom attendance already exists; do not claim the
    application category is novel.
2.  Lightweight face recognition is active research. FaceLiVT (2025) is
    a recent candidate; EdgeFace (2024) is a practical published
    comparison with public checkpoints.
3.  PAD is a separate task with separate datasets and metrics.
    Recognition accuracy alone is not anti-spoofing evidence.
4.  Generalisation to unseen cameras, lighting and attacks remains a
    concern.
5.  Privacy is part of the design: consent, local storage, deletion,
    pseudonymous exports and human review.

## Required literature matrix

For every paper actually cited in the final report, record:
`paper_id, citation, year, category, problem, method, dataset, protocol, metrics, reported_results, limitations, full_text_read (yes/no/partial), relevance, implementation_decision, DOI/URL`.

For each core paper read: abstract/introduction, method, experimental
protocol, datasets, results and limitations. Record whether full text or
only abstract/metadata was accessible. Never transfer a published
benchmark number into PRESENCIFY's results table.
