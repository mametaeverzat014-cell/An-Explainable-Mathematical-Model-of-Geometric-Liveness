# Repository audit and implementation plan

Date: 2026-10-08. Scope: turn the existing research project into an open,
explainable research instrument without altering the existing study.

## 1. Audit of the existing repository

**Structure.** A Python research pipeline (`src/`, `scripts/`, `tests/`,
`configs/prototype.yaml`), a report (`report.md`, `report.pdf`, Russian), a
poster, a protocol, a research log and committed aggregate results
(`results/*.csv`). Raw videos and per-frame landmarks are git-ignored. The new
work lives entirely in `web/` and does not touch these files.

**What is sound.** Subject-disjoint leave-one-subject-out evaluation; train-only
normalisation and thresholds (checked in `src/pls_model.py`); honest reporting
of the refuted hypothesis H2; the rule-of-three bound instead of “100 %”.

**Problems carried into the plan** (details were given in the review that
preceded this work):

1. *Authorship.* The git history shows that almost all code and text were
   produced by an AI assistant, while the report states the work was done
   independently. Addressed by `AI_ASSISTANCE.md` and by reserving the method
   for the students. *Changed on 2026-10-08: at the team's request the method
   was implemented by the assistant (`web/src/method/`); see AI_ASSISTANCE.md.*
2. *Ethics.* Faces of minors were recorded with no approval on record.
   Phase 2 is blocked until approval exists.
3. *Pseudo-depth.* The old features used MediaPipe's `z`, which is a learned
   prior that is present on photos too. The new method uses only 2D landmark
   positions.
4. *Sample size and attack coverage.* 6 participants, one attack species.
   Phase 2 plans a properly sized, pre-registered study.
5. *Replay attacks.* Proposition 7 of the report shows that 2D distance-ratio
   features cannot detect a replay. The new planarity test cannot either
   (MATH_SPEC.md, section 9). Phase 3 targets this.

## 2. Decisions

| Decision | Choice | Reason |
|---|---|---|
| Location | `web/`, existing files untouched | As requested. |
| Stack | TypeScript + Vite, no UI framework; Vitest; Playwright | Small dependency surface. Students can read every line. Runs on laptops and phones. |
| Method language | TypeScript, one implementation used by the app and the tests | A single source of truth. Phase 2 analysis scripts will reuse the same code from Node, so app and analysis cannot disagree. *If the students prefer Python, the specification is language-neutral; tell us before they start.* |
| Method | Homography fit + Sampson residuals + χ² test with calibrated σ | A standard, citable null model with explicit assumptions, replacing the uncalibrated variance score. |
| Privacy | Everything on-device; CSP `connect-src 'self'`; model and WASM self-hosted with a pinned checksum | No data can leave the device, and the property is verifiable. |
| Split of work | Originally: students M1–M7 and D1–D6. Since 2026-10-08: assistant implemented M1–M6; students own D1–D6, M7, experiments and interpretation | Authenticity of the scientific contribution; changed at the team's request. |
| Validation tests | Written against the specification; checked against a throwaway reference that is not committed | The tests must be passable and have realistic tolerances, without handing over a solution. |

## 3. Phases

### Phase 1 — planarity test instrument (this delivery)

- Infrastructure: linear algebra, χ² distribution, Wilson intervals, seeded
  RNG, synthetic scene built on MediaPipe's canonical face model, landmark
  subsets checked against the MediaPipe package.
- Capture: camera, MediaPipe Face Landmarker (VIDEO mode), per-frame
  timestamps (`requestVideoFrameCallback`), timing and sensor diagnostics.
- App: Live view (overlay, statistic, noise floor, provenance labels),
  Synthetic lab (single pair + Monte-Carlo size and power with Wilson
  intervals), Diagnostics, Method & status, About.
- Method interfaces M1–M6 with validation tests; MATH_SPEC.md. (Implemented 2026-10-08.)
- Tests: infrastructure unit tests, integration tests, Playwright smoke tests
  against the production build.

### Phase 2 — research mode

**Status (2026-10-09):** recording, export, offline and in-browser
analysis (with a bootstrap interval over recordings), an Experiment wizard
for Q4 and synthetic recordings are implemented (`docs/RESEARCH_MODE.md`).
The interface is translated into Russian and Kazakh. Recording people stays
blocked without an approval reference. Not yet built: protocol manager with a
frozen hash, and APCER/BPCER reporting per PAI species. Both depend on design
decisions the researchers have not made yet.

Original plan:

- Protocol manager: versioned, pre-registered protocol (hypotheses, sample
  size with justification, PAI species, motion script, landmark set, α,
  pair-selection rule), frozen by a hash before the first session.
- Consent flow with a recorded approval reference. Research mode refuses to
  start without it.
- Session recorder storing **landmark tracks and timestamps only** (no video)
  in memory, exported by the participant's operator as JSON/CSV. Pseudonymous
  IDs.
- Node analysis scripts reusing the same method code: per-presentation
  decisions, APCER per PAI species, BPCER, Wilson / Clopper–Pearson
  intervals, subject-level resampling.
- First experiments need no participants: flat prints only (Q4, Q1, Q3, Q5).

### Phase 3 — gyroscope consistency (smartphones)

- Record `DeviceMotionEvent` rotation rate with timestamps; measure camera ↔
  gyroscope latency (the diagnostics are prepared).
- Hypothesis to formalise: for a 3D face, the parallax direction and
  magnitude follow the phone's measured rotation. A replayed video cannot
  follow motion it never saw. Prior work exists (Li et al., CCS 2015), so the
  contribution must be positioned against it.

### Phase 4 — mathematical visualisations (optional)

Interactive derivations (plane-induced homography, parallax vs. depth),
built only from the validated synthetic scene.

## 4. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| A7 fails (MediaPipe imposes a 3D prior on photo landmarks) | χ² calibration invalid: false rejections on prints | Q4 is the first experiment. Fallback: empirical null distribution from prints. |
| Correlated landmark noise | p-values miscalibrated | Measure (Q1). Use an empirical null if needed. |
| Rolling shutter | False rejections grow with speed | Measure (Q5). Constrain motion speed in the protocol. |
| Students cannot complete M1–M7 in time | No working method | The specification is scoped to high-school linear algebra and statistics; the supervisor should check progress weekly. |
| Over-claiming | Credibility at judging | Provenance labels in the app; “Validated findings: none yet” until Phase 2. |
