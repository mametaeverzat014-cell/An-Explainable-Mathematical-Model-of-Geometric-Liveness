# Use of AI assistance in this project

This file records which parts of the repository were produced with an AI
assistant (Claude, by Anthropic) and which are the student authors' own work.
Keep it up to date, and reflect it in the report and at judging.

## Before 2026-10-08: the Python study

The git history (`git log`) shows that the Python pipeline (`src/`,
`scripts/`, `tests/`), the report (`report.md`, `report.pdf`), the poster,
the protocol, the research log and the presentation script were committed by
the AI assistant (24 of the first 25 commits carry `Co-Authored-By: Claude`).
The participant recordings and the run that produced `results/` were done by
a person (commit `881e6ac`).

The report's statement that the model, proofs, software and experiments were
done by the authors independently does not match this history. It must be
corrected before the report is submitted anywhere.

## From 2026-10-08: the web research instrument (`web/`)

Division of work, as agreed with the project lead and changed on 2026-10-08
(see “Change of plan” below):

| Part | Produced by |
|---|---|
| Application architecture, camera and MediaPipe integration, user interface, charts | AI assistant |
| Infrastructure numerics: matrix routines, eigen-solver, χ² distribution, Wilson interval, seeded RNG | AI assistant |
| Synthetic scene (canonical-face geometry, pinhole projection, noise) | AI assistant |
| Infrastructure tests, integration tests, browser smoke tests | AI assistant |
| Research-mode recorder, recording format, offline analysis and synthetic-recording scripts | AI assistant |
| Report correction list and judge-question list (`docs/`) | AI assistant (questions only, no answers) |
| Method specification and validation tests (`web/docs/MATH_SPEC.md`, `web/src/method/tests/`) | AI assistant |
| **Implementation of the method, M1–M6** (`web/src/method/*.ts`) and its explanation (`web/docs/METHOD_EXPLAINED_RU.md`) | **AI assistant**, at the team's request |
| Derivations D1–D6, the motion threshold M7, understanding and defending the method | Student authors |
| Open-question experiments, interpretation, report | Student authors |

**Change of plan (2026-10-08).** The method was originally reserved for the
student authors, and `web/src/method/` (then `web/src/student/`) contained
only empty stubs. On 2026-10-08 the person coordinating the project for the
team asked the assistant to implement it, because the authors did not know
how to start. The assistant implemented M1–M6, renamed the folder so that its
name no longer suggests student authorship, and wrote an explanation in
Russian. The scientific contribution that remains with the authors is the
derivations, the motion-threshold analysis, the experiments and their
interpretation. Any report or presentation must describe the method's code
as AI-written.

**Checking the tests.** To make sure every validation test can be passed and
that its tolerances are realistic, the assistant wrote a separate reference
implementation of M1–M6 outside the repository before the change of plan.
It was used to run the test suite, to take screenshots and once to run
`npm run analyze` end to end on synthetic recordings. The committed
implementation is that same code with explanatory comments. The simulation
numbers quoted in `MATH_SPEC.md` come from it; the authors should reproduce
them in the Synthetic lab.

**Literature.** The reference list in `MATH_SPEC.md` was written from the
assistant's memory. Every citation must be checked against the original
publication before use.

## Rules going forward

- Commits made with AI help keep the `Co-Authored-By` trailer.
- Work the authors do themselves (derivations, M7, experiments, analysis
  choices) is committed or documented under their own names.
- Any further AI help is recorded in this file: what was asked and what was
  produced.
