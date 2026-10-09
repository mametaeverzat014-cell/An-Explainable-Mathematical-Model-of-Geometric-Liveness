# Mathematical specification — planarity test

This document specifies the research method of Parallax Lab.

**Status (2026-10-08).** M1–M6 are implemented in `web/src/method/` by the AI
assistant at the team's request (`AI_ASSISTANCE.md`), and pass the validation
suite (`npm run test:method`). A plain-language walkthrough in Russian is in
`METHOD_EXPLAINED_RU.md`. The tasks below remain the authors' work, as
**understanding and derivation** tasks:

1. Do the derivation tasks (D1–D6) on paper. Keep the pages: they are part of
   your research record and what judges will ask about.
2. For each of M1–M6, read the implementation and be able to explain every
   line, and why the tests check what they check.
3. Then work on M7 and the open questions. These are research, not homework:
   nobody knows the answers yet.

Nothing in this document is a finished result. Statements marked
**(theorem)** are standard results with a reference. Statements marked
**(to verify)** are claims you must check. Statements marked
**(open)** are unknown.

---

## 1. Notation

- An image point is `x = (x, y)` in pixels; its homogeneous form is
  `x̃ = (x, y, 1)ᵀ`. Two homogeneous vectors represent the same point if one is
  a non-zero multiple of the other.
- A **homography** is an invertible 3×3 matrix `H`, defined up to scale
  (8 degrees of freedom). It maps `x̃ ↦ H x̃`. In code, `H` is a length-9 array,
  row-major: `[h11, h12, h13, h21, h22, h23, h31, h32, h33]`.
- A frame pair gives `n` correspondences `xᵢ ↔ x′ᵢ`: the same MediaPipe
  landmark in an earlier frame (`src`) and the current frame (`dst`).

## 2. The question, as hypotheses

**H₀ (planar).** There is a homography `H` and a noise level `σ` such that

    xᵢ = x̄ᵢ + εᵢ,   x′ᵢ = x̄′ᵢ + ε′ᵢ,   x̄′ᵢ ~ H x̄ᵢ   for all i,

where all error coordinates are independent `N(0, σ²)`.

**H₁ (non-planar).** No such `H` exists: the true points `x̄′ᵢ` cannot all be
explained by one homography.

The test answers *“is H₀ plausible for this frame pair?”*. It does not answer
*“is this a live person?”*. Section 9 lists everything besides a 3D face that
can make H₀ false.

## 3. Why a plane gives a homography

**D1 (theorem; derive it yourself).** Let a plane π satisfy `nᵀX = d` in the
first camera's coordinates, and let the second view be related by rotation `R`
and translation `t`. With intrinsic matrices `K`, `K′`, show that images of
points on π satisfy `x̃′ ~ H x̃` with

    H = K′ (R − t nᵀ / d) K⁻¹.

Reference: Hartley & Zisserman, *Multiple View Geometry in Computer Vision*,
2nd ed., section 13.1. Write out every step, including where `nᵀX = d` is used.

A moving flat print in front of a fixed camera is the same situation: the
relative motion between camera and plane is what matters.

**D2 (to verify).** Under scaled orthographic projection, show that a planar
object maps between views by an *affine* transform. Then take a point at
distance `δ` in front of the plane and rotate the object by `θ` about a
vertical axis. Show that the point's image moves by approximately
`s · δ · sin θ` (s = image scale, px per cm) *more* than a plane point would.
This is the parallax the test detects. Use it to predict the residual of the
nose tip of the canonical face (`δ ≈ 4 cm` relative to the eye corners) for
`θ = 5°` at 50 cm with `f = 1000 px`, and compare with the Synthetic lab
(single-pair view, σ = 0). Record the prediction **before** you look.

## 4. Method tasks

### M1. Normalised DLT — `estimateHomography(src, dst)`

**D3 (derive).** From `x̃′ × (H x̃) = 0`, derive the two linear equations in
the nine entries of `H` contributed by each correspondence (H&Z eq. 4.3).
Stack them as `A h = 0` with `A` of size 2n × 9. Explain why `h` is chosen
as the unit vector minimising `‖A h‖`, and why at least 4 correspondences in
general position are needed.

Algorithm (H&Z Algorithm 4.2):

1. Normalise each point set separately: translate so the centroid is at the
   origin, then scale so the mean distance from the origin is √2. Write this
   as a 3×3 matrix `T` (for src) and `T′` (for dst).
2. Build `A` from the normalised points.
3. `h̃` = unit vector minimising `‖A h‖`. Infrastructure:
   `smallestRightSingularVector(rows)` from `core/linalg.ts`. It also returns
   the second-smallest eigenvalue: if that is ≈ 0, the solution is not unique
   (degenerate configuration). Decide on a threshold and justify it.
4. De-normalise: `H = T′⁻¹ H̃ T` (`mat3Inv`, `mat3Mul`).

**D4 (experiment).** Implement a version *without* step 1 temporarily, and
measure transfer error for noisy points at coordinates around 2000 px. Explain
the difference using condition numbers (Hartley, 1997). Keep the numbers.

Contract: see the comment in `homography.ts`; tests in
`src/method/tests/homography.test.ts`.

### M2. Sampson error — `sampsonErrorsSquared(H, src, dst)`

For one correspondence, let `ε(x, y, x′, y′)` be the 2-vector of algebraic
errors from M1 (the two rows of `A` times `h`) and `J = ∂ε / ∂(x, y, x′, y′)`
(2×4). The squared Sampson error is

    eᵢ² = εᵀ (J Jᵀ)⁻¹ ε.

It is the first-order approximation of the squared distance the 4-vector
`(x, y, x′, y′)` must move to satisfy `x̃′ ~ H x̃` exactly (H&Z section 4.2.6).

**D5 (derive).** Write `ε` and all eight entries of `J` explicitly. Then work
the check cases below by hand; the tests use exactly these.

- *Check case 1.* `H = I`, `x′ = x + d`: then `e² = ‖d‖² / 2`.
  Hint: for `H = I` the constraints are linear in the coordinates, so the
  first-order approximation is exact, and the correction is split equally
  between the two images.
- *Check case 2.* `H = diag(s, s, 1)`: then
  `e² = [(s x − x′)² + (s y − y′)²] / (1 + s²)`.
- *Check case 3.* If `H` is exact and both points carry `N(0, σ²)` noise,
  `E[e²] ≈ 2σ²`. Explain why it is 2σ² and not σ²: count the directions
  normal to the constraint surface in ℝ⁴.
- `e²` must not change when `H` is multiplied by a non-zero constant. Prove it.

### M3. Test statistic — `planarityStatistic(errorsSquared, sigmaPx)`

    C = Σᵢ eᵢ²,      T = C / σ²,      dof = 2n − 8,      p = P(χ²_dof ≥ T).

**(theorem, asymptotic)** If `H` were the maximum-likelihood estimate and
`C` the exact geometric cost, `T` would follow approximately `χ²` with
`N − d` degrees of freedom, where `N = 4n` measured coordinates and
`d = 2n + 8` estimated parameters (n corrected points plus 8 for `H`). See
H&Z chapter 5 (evaluation of estimators, residual error).

**D6 (to verify).** Your implementation uses the DLT estimate and the Sampson
approximation, not the exact ML solution. Explain why `T` should still be
close to `χ²(2n − 8)` for small noise, then check it: the Monte-Carlo test in
`tests/monte-carlo.test.ts` requires the false-rejection rate on a flat print
to lie in [0.03, 0.07] at α = 0.05. Report the rate you get and its
uncertainty (Wilson interval, `wilsonInterval` in `core/stats.ts`).

Infrastructure you may use: `chiSquareSf(T, dof)`.

### M4. Decision rule — `decide({ stat, alpha, motionDeg, minMotionDeg })`

Three outcomes, checked **in this order**:

1. `stat === null` → *inconclusive* (no noise calibration, or a stage failed).
2. `motionDeg < minMotionDeg` → *inconclusive*, **whatever the p-value**.
3. `p < α` → *non-planar*; otherwise *planar-consistent*.

Why the order matters: with too little motion even a 3D face is almost a
plane, so the test has no power, and a non-rejection means nothing. The gate
must be decided from motion alone. If it looked at the p-value too, the rule
could be tuned to produce the answer one hopes for.

The `reason` string is shown to users. It must say *why*, in one sentence,
and must never say “live”, “real” or “attack”.

### M5. Noise estimate — `estimateNoiseSigma(pairs)`

During calibration the presented object is held still, so (approximately)
H₀ holds for every calibration pair `k`. Fit `H_k`, compute
`C_k = Σᵢ e²ᵢ,k`, and pool:

    σ̂² = Σₖ C_k / Σₖ (2nₖ − 8).

Show that `E[σ̂²] = σ²` under the M3 approximation. Skip pairs with fewer than
5 correspondences. Return `{ sigmaPx: σ̂, dof: Σₖ(2nₖ − 8) }`.

**Caution (open, Q3).** Calibration pairs from consecutive frames may share
errors (MediaPipe smooths landmarks over time). That would make `σ̂` too
small, and too many moving prints would be called non-planar. The app uses
pairs 5 frames apart (a placeholder, not a derived value).

### M6. Confidence interval for σ — `sigmaConfidenceInterval(estimate, confidence)`

If `ν σ̂² / σ² ~ χ²_ν` with `ν = dof`, derive a two-sided interval for `σ`
with the requested coverage (use `chiSquareQuantile`). The tests check
coverage by simulation: 1000 replications must cover the true σ between 93 %
and 97 % of the time for a 95 % interval.

### M7. Power analysis — replaces the placeholder `minMotionDeg = 5°`

The minimum motion should be **derived**: the smallest rotation for which a
3D face is detected with power ≥ 0.8 at α = 0.05, given the measured `σ` and
the face size in pixels (outer-eye-corner distance).

Suggested route:

1. Using D2, approximate the expected parallax of each rigid landmark as a
   function of rotation, face size and that landmark's depth relief.
2. Under H₁, `T` is approximately non-central `χ²` with non-centrality
   `λ ≈ Σᵢ (parallax not absorbed by H)ᵢ² / (2σ²)` (to verify).
3. Compute the rotation at which `P(T > χ²_{1−α}) = 0.8`. Check against the
   Synthetic lab's Monte-Carlo sweep. Present both: the formula and the
   simulation.

Indicative simulation numbers (yaw, 25 rigid landmarks, 50 cm, f = 1000 px):
power ≈ 0.30 / 0.97 / 1.0 at 1° / 2° / 3° for σ = 0.5 px, but only
≈ 0.18 / 0.49 at 3° / 5° for σ = 2 px, and ≈ 0.10 at 3° when the distance is
100 cm with σ = 1 px. **Power depends strongly on noise and face size.** A
fixed 5° gate is therefore not justified.

## 5. Pair selection (infrastructure; understand it)

Among the frames of the last `windowMs`, the reference frame is the one whose
MediaPipe head pose differs most from the current frame. The rule looks only
at motion, never at residuals, so it does not bias the test.

The live display tests overlapping, strongly dependent frame pairs about
10 times per second. Its stream of p-values is **not** a set of independent
tests. A per-recording decision (Phase 2) needs one pre-specified pair (or a
pre-specified way of combining pairs that accounts for dependence). Taking
the smallest p-value of a recording is not allowed.

## 6. The synthetic scene (infrastructure; know its limits)

- Geometry: the 468 vertices of MediaPipe's canonical face model (cm), or the
  same face printed flat (`z = 0`), or bent around a vertical cylinder.
- Camera: ideal pinhole, `f = 1000 px`, 1280×720. No lens distortion, no
  rolling shutter, no motion blur.
- Noise: independent `N(0, σ²)` on every coordinate, with **known** σ.
- Motion: rotation about a pivot 4 cm behind the canonical origin (face) or at
  the sheet centre (prints).

Every synthetic number is a property of the method *under these
assumptions*. None of it is evidence about real cameras or real people.

## 7. Open questions — falsifiable predictions

Each question states what the idealised theory predicts and how to test it
with **flat prints held by the researchers themselves**. That data is
non-human data, but confirm with your supervisor whether your own faces on
the prints need approval.

| | Question | Prediction if the assumptions hold | How to test |
|---|---|---|---|
| **Q4** (first) | Do MediaPipe's 2D landmarks on a tilted *photo* follow the photo, or does the network “imagine” a 3D head? (assumption A7) | Moving flat print: T/dof ≈ 1, false-rejection rate ≈ α | Rotate a rigidly mounted print by known angles; compare rejection rate with α (Wilson interval). If far above α even with careful calibration, A7 fails and the method needs an empirical null distribution. |
| Q1 | Are landmark errors independent Gaussian? (A5) | Still flat print: histogram of T matches χ²(2n−8) | Kolmogorov–Smirnov test against χ²; plot the empirical CDF. Compare `rigid` vs `features` sets. |
| Q3 | Does noise grow with motion? (A6) | σ̂ from moving-print pairs ≈ σ̂ from still pairs | Estimate σ from moving-print pairs with M5 (H₀ holds for prints) and compare intervals. |
| Q5 | Rolling shutter (A2) | False-rejection rate independent of angular speed | Rotate the print at controlled speeds; regress rejection on speed. |
| Q2 | Which landmark set? | Rigid set: fewer violations of A3 | Compare size (prints) and power (later, faces) across sets; fix the choice **before** Phase 2. |
| Q6 | Face size | Power rises with face size in px | Prints at several distances (size), simulation (power). |

## 8. Positioning against prior work

The planarity / parallax idea is **not new**. The contribution can be an
open, explainable and statistically calibrated version, with honest
measurement of its assumptions. You must read and cite at least these.
Entries marked ✓ were confirmed by a web search on 2026-10-09 (DOI or
publisher record); the method descriptions come from abstracts, because full
texts could not be downloaded. A fuller list with links and a Russian
discussion is in `docs/PRIOR_WORK_RU.md`.

- ✓ R. Hartley, A. Zisserman. *Multiple View Geometry in Computer Vision*,
  2nd ed., Cambridge University Press, 2003/2004. Homography estimation, DLT,
  Sampson error, evaluation, planes and homographies (check chapter numbers
  against the book).
- ✓ R. Hartley. *In defense of the eight-point algorithm.* IEEE TPAMI
  19(6):580–593, 1997 (why normalisation matters).
- ✓ P. H. S. Torr. *Bayesian model estimation and selection for epipolar
  geometry and generic manifold fitting.* IJCV 50(1):35–61, 2002 (GRIC:
  plane vs. general 3D model selection); K. Kanatani. *Geometric information
  criterion for model selection.* IJCV 26(3):171–189, 1998. The plane-vs-3D
  decision from residuals is standard; the project does not claim a new test.
- M. Irani, P. Anandan. *Parallax geometry of pairs of points for 3D scene
  analysis.* ECCV 1996 (plane + parallax). Not re-checked.
- ✓ M. De Marsico, M. Nappi, D. Riccio, J.-L. Dugelay. *Moving face spoofing
  detection via 3D projective invariants.* ICB 2012, pp. 73–78,
  doi:10.1109/ICB.2012.6199761. **Closest prior work: uses projective
  invariants (cross-ratios) of facial points to detect planar spoofs, without
  training.** Your report must explain how your test differs (explicit noise
  model, calibrated p-values, measured assumptions).
- ✓ T. Wang, J. Yang, Z. Lei, S. Liao, S. Z. Li. *Face liveness detection using
  3D structure recovered from a single camera.* ICB 2013.
- ✓ W. Bao, H. Li, N. Li, W. Jiang. *A liveness detection method for face
  recognition based on optical flow field.* IASP 2009 (via survey citations).
- ✓ K. Kollreider, H. Fronthaler, J. Bigun. *Non-intrusive liveness detection by
  face images.* Image and Vision Computing 27(3):233–244, 2009,
  doi:10.1016/j.imavis.2007.05.004.
- ✓ J. H. Connell, N. K. Ratha (IBM). *Spoof detection for facial
  recognition.* US Patent 9,898,674 B2 (displacements of facial points
  tested against a 3D surface model).
- ✓ Y. Li, Y. Li, Q. Yan, H. Kong, R. H. Deng. *Seeing your face is not
  enough: an inertial sensor-based liveness detection for face
  authentication.* ACM CCS 2015, doi:10.1145/2810103.2813612; K. T. Nguyen et
  al., *Face spoofing detection for smartphones using a 3D reconstruction and
  the motion sensors*, ICISSP 2018 (both relevant to Phase 3).
- ✓ Y. Xu et al. *Virtual U: defeating face liveness detection by building
  virtual models from your public photos.* USENIX Security 2016 (an attack a
  planarity test cannot detect).
- ✓ Y. Kartynnik et al. *Real-time facial surface geometry from monocular
  video on mobile GPUs.* CVPR Workshops 2019, arXiv:1907.06724 (MediaPipe
  Face Mesh).
- ✓ ISO/IEC 30107-3, *Biometric presentation attack detection — Part 3:
  Testing and reporting.* The 2017 edition was withdrawn on 2023-01-10; the
  current edition is ISO/IEC 30107-3:2023. Use its vocabulary: *bona fide
  presentation*, *attack presentation*, *presentation attack instrument
  (PAI)*, *PAI species*, *APCER* (computed **per PAI species**), *BPCER*.
  ACER is not one of the standard's metrics; it comes from competition
  protocols.

Learning resources:

- R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., 2022
  (free online): image formation, feature-based alignment.
- G. Strang, *Introduction to Linear Algebra*: least squares, eigenvectors,
  SVD.
- Any introductory statistics text: χ² distribution, hypothesis tests, type I
  and type II error, power, binomial confidence intervals.

## 9. What the test cannot do

- **Replay attacks.** A video replayed on a screen contains real head motion
  that was recorded earlier, so its landmarks are not planar-consistent. This
  follows from the method, not from a lack of data. Phase 3 addresses it.
- **3D masks, curved prints, bent paper.** These are non-planar. The test may
  reject planarity for them exactly as for a real face.
- **Expressions and blinks** make landmarks move non-rigidly, which also
  rejects H₀. The rigid landmark set reduces this but cannot remove it.
- **No motion → no information.** A face held perfectly still is
  indistinguishable from a photo.

So “non-planar” never means “live”, and “planar-consistent” never means
“attack”. In ISO/IEC 30107-3 terms, the planarity test could at most become
one component of a PAD subsystem for the PAI species “flat print” and “still
image on a screen”. Its APCER and BPCER for those species are unknown until
Phase 2.

## 10. Verifying

```
cd web
npm run test:method         # validation suite for M1–M6
npm test                    # infrastructure tests (must always pass)
```

The validation suite was written before the implementation, from this
specification, and checked against a throwaway reference to make sure its
tolerances are realistic. The committed implementation is that reference with
comments. The `METHOD_IMPL_DIR` environment variable lets a supervisor run the
same tests against an independent implementation.
