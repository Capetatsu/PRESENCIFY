# Agent Rules --- SmartAttend

## Before coding

1.  Inspect the repository and report its structure/current run
    commands.
2.  Read every Markdown file in this pack.
3.  Create a task checklist and implement the smallest first milestone.
4.  Preserve working architecture; explain any proposed rewrite before
    doing it.

## Engineering rules

-   Target Windows and CPU-only inference; no CUDA requirement.
-   Do not train large models from scratch.
-   Isolate model smoke tests and do not install every research-repo
    dependency blindly.
-   Load models once, not per frame. Keep UI responsive with controlled
    frame rate/resize.
-   Make thresholds/configuration explicit; no hidden magic values.
-   Log model errors, checkpoint absence, device and latency.
-   Add tests for session rules and database uniqueness.
-   No fake predictions in the live view. Synthetic demo data must be
    clearly labelled.
-   Never silently download large datasets or weights; state what is
    needed.
-   Do not delete files or rewrite architecture without inspecting it.

## Research rules

-   Core references in the submitted paper must be published 2024--2026.
-   Verify title, authors, final publication year, venue and
    DOI/official URL.
-   Distinguish publication year from preprint year and from the year of
    a backbone referenced by a paper.
-   Do not invent citations, metrics, sample sizes, weights or code
    availability.
-   A paper's benchmark numbers are not SmartAttend results.
-   If a paper's method has no accessible runnable code/checkpoint, mark
    it "reviewed candidate, not implemented."
-   The paper must name the model/checkpoint the code actually uses.
-   A blink/head-turn challenge is not equivalent to validated PAD.

## Privacy and decisions

-   Use consenting participants; keep data local by default.
-   Support deleting templates/records.
-   Export pseudonymous experiment data; avoid raw frames in logs.
-   Route uncertain decisions to teacher review.
-   Explain limits of face recognition and PAD.

## Milestone report

After each milestone, report changed files, exact run command, tests and
actual results, working features, unverified/broken parts and next step.
Never say everything works unless acceptance tests passed.
