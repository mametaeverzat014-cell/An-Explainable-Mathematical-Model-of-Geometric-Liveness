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

Division of work agreed with the project lead:

| Part | Produced by |
|---|---|
| Application architecture, camera and MediaPipe integration, user interface, charts | AI assistant |
| Infrastructure numerics: matrix routines, eigen-solver, χ² distribution, Wilson interval, seeded RNG | AI assistant |
| Synthetic scene (canonical-face geometry, pinhole projection, noise) | AI assistant |
| Infrastructure tests, integration tests, browser smoke tests | AI assistant |
| Method specification and validation tests (`web/docs/MATH_SPEC.md`, `web/src/student/tests/`) | AI assistant |
| **The method: `web/src/student/*.ts` (tasks M1–M7), derivations D1–D6** | **Student authors** |
| Open-question experiments, interpretation, report | Student authors |

The interfaces in `web/src/student/` were written by the assistant as empty
stubs that throw `NotImplementedError`.

**Checking the tests.** To make sure every validation test can be passed and
that its tolerances are realistic, the assistant wrote a separate reference
implementation of M1–M6 outside the repository. It was used only to run the
test suite and to take screenshots of the interface, and it was not
committed. Two simulation results quoted in `MATH_SPEC.md` (test size close
to α; indicative power numbers) come from that check. The students should
reproduce them with their own implementation and report their own numbers.

**Literature.** The reference list in `MATH_SPEC.md` was written from the
assistant's memory. Every citation must be checked against the original
publication before use.

## Rules going forward

- Commits made with AI help keep the `Co-Authored-By` trailer.
- Student-owned files (`web/src/student/*.ts`, excluding `tests/`) are
  committed by the students under their own names.
- If a student asks the assistant for help with an M-task, record what was
  asked and what was answered here.
