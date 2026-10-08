# Research mode: recording and analysis

Infrastructure for Phase 2: record landmark tracks in the browser, download
them as files, analyse them offline with the same method code the app uses.

**What is not decided here.** The experimental design (hypotheses,
conditions, sample sizes, motion script, landmark set, α, pair rule, unit of
analysis) belongs to the researchers and must be fixed in a protocol
**before** data are collected. The tools take these as parameters.

## 1. What a recording contains

One JSON file per recording (`src/core/recording.ts`, format version 1):

- `meta`: random ID, date, subject category, role (`calibration` or
  `trial`), condition, protocol ID, free-text notes, camera resolution and
  frame rate, timestamp source, number of frames without a face. For a person:
  pseudonym and ethics approval reference.
- `frames`: for every frame with a face, the time since the first frame (ms),
  MediaPipe's 3×3 head rotation, and the 468 landmark positions in pixels,
  rounded to 0.01 px.

No video, no images, no names. Landmark tracks of a **real person** are still
derived biometric data, so treat those files as confidential: store them only
where the approval says, and do not commit them to git (`recordings/` is
git-ignored).

## 2. Safeguards

- The app records a person only when an approval reference, a pseudonym and a
  consent confirmation are entered. The analysis refuses human-participant
  files without an approval reference (`validateRecording`).
- Recordings stay in the browser tab's memory until **Download JSON** is
  pressed. Nothing is uploaded; the site cannot connect to other servers.
- File names contain date, pseudonym or “target”, condition, role and ID only.

An approval reference that is typed in is not checked against any registry.
The safeguard prevents accidents; it cannot prevent misuse. Honesty about
approvals is the researchers' responsibility.

## 3. Recording workflow (prints held by the researchers)

1. Open **Record**, start the camera.
2. Subject: “A print or screen held by the researchers”. Enter the protocol
   ID and describe the setup in Notes.
3. **Calibration**: hold the print still for the set duration (default 3 s).
   Download the file.
4. **Trials**: move the print as the protocol prescribes. Download each file.
5. Repeat the calibration whenever camera, distance or lighting change.

## 4. Analysis

```bash
cd web
npm run analyze -- --calibration recordings/<calibration>.json \
                   --set rigid --window 1500 --min-motion 5 --alpha 0.05 \
                   --out analysis-out recordings/<trial1>.json recordings/<trial2>.json
```

- σ is estimated from the calibration file with the students' M5 function.
  The calibration is rejected if the pose changed by more than
  `--max-cal-motion` degrees.
- **Pair rule “non-overlapping windows”**: each trial is cut into consecutive
  windows of `--window` ms; in each window the first frame is paired with the
  frame of largest MediaPipe rotation from it. No frame is used twice. The
  rule looks only at motion.
- Pairs below `--min-motion` are reported but not counted as tested.
- Output: `pairs.csv` (one row per pair: motion, n, T, dof, p, outcome) and
  `summary.json` (settings, σ with its interval, per-recording rejection rate
  with a Wilson 95 % interval).

Until the method (`src/student/`) is implemented, the analysis runs and
reports which stage is missing, without inventing numbers.

**Statistical caution.** Pairs from the same recording are not independent
(same object, same session, temporally correlated landmark noise). The Wilson
interval treats them as independent and is therefore too narrow. The protocol
must define the unit of analysis, for example one pre-specified pair per
recording, or many recordings per condition with the recording as the unit.

## 5. Trying it without real data

```bash
npm run make-synthetic -- recordings/synthetic
npm run analyze -- --calibration recordings/synthetic/*calibration*.json \
                   --out analysis-out recordings/synthetic/*trial*.json
```

This writes a still calibration and three trials (flat print, 3D face, curved
print) from the synthetic scene, in the real file format.

## 6. Suggested first experiment (no participants)

This tests open question Q4 in `MATH_SPEC.md`: does MediaPipe treat a moving
flat print as a plane? It is a suggestion; the researchers design and
pre-register the actual protocol.

- **Prediction under the method's assumptions:** for flat prints, the
  false-rejection rate is close to α.
- **Falsified if** the rate is clearly above α, with the interval computed at
  the level of independent recordings, across repeated sessions.
- **Things to fix in advance:** number of recordings, motion (angle, speed),
  distance, lighting, camera, landmark set, window, motion gate, α, and how
  recordings are combined.
