# First Prompt to Give Your Vibecoding Agent

You are the implementation agent for **PRESENCIFY**, a deadline-driven
research prototype. Read every Markdown file in this folder before
coding, especially `PROJECT_CONTEXT.md`, `RESEARCH_REVIEW_2024_2026.md`,
`IMPLEMENTATION_PLAN.md`, `EXPERIMENTS_AND_PAPER.md`, and
`AGENT_RULES.md`.

## Context

I need a genuinely working Windows laptop webcam demo and a defensible
research paper before 15 October 2026. My machine has an Intel i3-class
CPU, 8 GB RAM, Intel UHD integrated graphics, and no NVIDIA GPU. Do not
assume CUDA. I am vibecoding the project, so implement in small
verifiable milestones and keep the app runnable.

## Your first task --- do not build everything at once

1.  Inspect the entire current repository and report its structure,
    framework, entry points, run commands, dependencies, broken parts
    and current git state.
2.  Read the supplied project Markdown files and reconcile the spec with
    the actual code.
3.  Create `docs/IMPLEMENTATION_STATUS.md` with a checklist and risks.
4.  Create an isolated CPU smoke-test plan for the face detector,
    FaceLiVT (2025) recognition candidate, EdgeFace (2024)
    comparison/fallback, and the 2025/2026 PAD candidates.
5.  Verify exact official code, usable checkpoint, license,
    preprocessing, model input shape and CPU inference before selecting
    a model. Do not assume that a paper's code or weights are available.
6.  Implement only the first smallest milestone: a reliable webcam test
    and a minimal project run command. Add a test/checklist and report
    the exact command and actual result.
7.  Stop and report what passed, what failed, and the next step. Do not
    implement later phases until the first milestone is verified.

## Non-negotiable

-   No fabricated results, citations, model capabilities or test status.
-   Do not label a similarity score as a probability.
-   Do not claim liveness/PAD works unless a real PAD model ran.
-   Do not fake predictions in the live UI.
-   Use consenting participants and pseudonymous experiment data.
-   Every paper claim must be supported by a verified 2024--2026 source,
    and every experimental number must come from actual logs.
