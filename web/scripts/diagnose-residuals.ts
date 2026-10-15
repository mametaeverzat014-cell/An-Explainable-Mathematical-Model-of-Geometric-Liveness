// Exploratory diagnostics for flat-print recordings (open question Q4):
// when the planarity test rejects a flat print, is the cause random landmark
// noise that the still-hold calibration underestimates, or a systematic,
// parallax-like pattern (MediaPipe placing landmarks as if the photo had a
// 3D face's relief)?
//
//   npm run diagnose -- <bundle-or-recording files...>
//
// Each file is analysed with the calibration recording(s) it contains. Uses
// the app's own pair rule (non-overlapping windows), landmark set and method.
//
// These measures were designed AFTER seeing the first real data (14 Oct
// 2026). They are exploratory, not part of the pre-registered test; report
// them as such.
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { correspondences, interocularPx } from '../src/core/frames';
import { LANDMARK_SETS } from '../src/core/landmark-sets';
import { analyzePair } from '../src/core/method';
import { framesOf, parseRecordingsFile } from '../src/core/recording';
import { calibrationFromRecording, DEFAULT_SET_SETTINGS as S, selectWindowPairs } from '../src/core/recording-analysis';
import { Rng } from '../src/core/rng';
import { CANONICAL_FACE, makeObject, projectObject, syntheticPair } from '../src/core/synthetic';
import type { Point2 } from '../src/core/types';

const { positionals } = parseArgs({ allowPositionals: true, options: {} });
if (!positionals.length) {
  console.error('usage: npm run diagnose -- <files...>');
  process.exit(1);
}

const idx = LANDMARK_SETS[S.landmarkSet].indices;
const CHI2_95_42 = 58.124; // χ²(42) 0.95 quantile; 25 landmarks → dof 42

// --- Two candidate "parallax" patterns over the rigid landmarks -------------
// (a) face relief: each landmark's height above the best plane through the
//     canonical 3D face (z regressed on x, y), in cm;
// (b) a sheet bent around a vertical axis: height ~ x².
function centred(v: number[]): number[] {
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return v.map((x) => x - m);
}
function reliefAbovePlane(): number[] {
  const P = idx.map((i) => CANONICAL_FACE[i]);
  // Normal equations for z = a x + b y + c, solved by Cramer's rule.
  let sxx = 0, sxy = 0, syy = 0, sx = 0, sy = 0, sxz = 0, syz = 0, sz = 0;
  for (const p of P) {
    sxx += p.x * p.x; sxy += p.x * p.y; syy += p.y * p.y; sx += p.x; sy += p.y;
    sxz += p.x * p.z; syz += p.y * p.z; sz += p.z;
  }
  const A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, P.length]];
  const B = [sxz, syz, sz];
  const det = (m: number[][]) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(A);
  const [a, b, c] = [0, 1, 2].map((k) => det(A.map((row, r) => row.map((v, col) => (col === k ? B[r] : v)))) / D);
  return P.map((p) => p.z - (a * p.x + b * p.y + c));
}
const relief = centred(reliefAbovePlane());
const bend = centred(idx.map((i) => CANONICAL_FACE[i].x ** 2));

/** Least-squares fit r_i = w_i · v: returns the share of residual energy explained and |v|. */
function patternFit(res: readonly Point2[], w: readonly number[]): { share: number; v: number } {
  let ww = 0, wx = 0, wy = 0, e = 0;
  res.forEach((r, i) => {
    ww += w[i] * w[i]; wx += w[i] * r.x; wy += w[i] * r.y; e += r.x * r.x + r.y * r.y;
  });
  return { share: (wx * wx + wy * wy) / ww / e, v: Math.hypot(wx, wy) / ww };
}

// Synthetic 3D face, noise-free, turned by `deg` (yaw) at the distance that
// gives the same face size in px: the relief signal a REAL face would produce.
const iodAt50 = (() => {
  const p = projectObject(makeObject('face3d'), { yawDeg: 0, pitchDeg: 0 }, 50);
  return Math.hypot(p[33].x - p[263].x, p[33].y - p[263].y);
})();
function realFaceRelief(deg: number, facePx: number): number {
  const d = (50 * iodAt50) / facePx;
  const c = syntheticPair({ object: makeObject('face3d'), pose0: { yawDeg: -deg / 2, pitchDeg: 0 }, pose1: { yawDeg: deg / 2, pitchDeg: 0 }, distanceCm: d, sigmaPx: 0, indices: idx }, new Rng(0));
  const a = analyzePair(c, { sigmaPx: 1, alpha: 0.05, minMotionDeg: 0, motionDeg: deg });
  return patternFit(a.residualVectors!, relief).v;
}

const med = (v: number[]) => (v.length ? [...v].sort((x, y) => x - y)[Math.floor(v.length / 2)] : NaN);
function spearman(a: number[], b: number[]): number {
  const rank = (v: number[]) => {
    const o = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array<number>(v.length);
    o.forEach(([, i], k) => (r[i] = k));
    return r;
  };
  const ra = rank(a), rb = rank(b), n = a.length;
  const d2 = ra.reduce((s, x, i) => s + (x - rb[i]) ** 2, 0);
  return 1 - (6 * d2) / (n * (n * n - 1));
}

interface Row { file: string; motion: number; facePx: number; rmsPx: number; tOverDof: number; relief: number; bend: number; kappa: number; tested: boolean; res: Point2[]; C: number }
const rows: Row[] = [];

for (const file of positionals) {
  const recs = parseRecordingsFile(readFileSync(file, 'utf8'));
  const cals = recs.filter((r) => r.meta.role === 'calibration');
  if (!cals.length) {
    console.error(`${file}: no calibration recording, skipped`);
    continue;
  }
  const cal = calibrationFromRecording(cals[0], idx, S.calibrationGap, S.maxCalibrationMotionDeg);
  if (!cal.estimate) {
    console.error(`${file}: calibration failed (${cal.message}), skipped`);
    continue;
  }
  const sigma = cal.estimate.sigmaPx;
  // Label rows by protocol ID when there is one (file names are often generic).
  const name = recs[0].meta.protocolId ?? file.split('/').pop()!;
  console.log(`${name}: calibration σ = ${sigma.toFixed(3)} px [${cal.interval95!.map((v) => v.toFixed(3)).join(', ')}]`);
  for (const r of recs.filter((x) => x.meta.role === 'trial')) {
    for (const p of selectWindowPairs(framesOf(r), { windowMs: S.windowMs })) {
      const a = analyzePair(correspondences(p.ref, p.cur, idx), { sigmaPx: sigma, alpha: S.alpha, minMotionDeg: S.minMotionDeg, motionDeg: p.motionDeg });
      if (!a.stat || !a.residualVectors) continue;
      const C = a.stat.T * sigma * sigma;
      const facePx = interocularPx(p.ref);
      const rf = patternFit(a.residualVectors, relief);
      rows.push({
        file: name, motion: p.motionDeg, facePx, rmsPx: Math.sqrt(C / a.stat.dof), tOverDof: a.stat.T / a.stat.dof,
        relief: rf.share, bend: patternFit(a.residualVectors, bend).share,
        kappa: p.motionDeg > 0.5 ? rf.v / realFaceRelief(p.motionDeg, facePx) : NaN,
        tested: p.motionDeg >= S.minMotionDeg, res: a.residualVectors, C,
      });
    }
  }
}

const tested = rows.filter((r) => r.tested);
console.log(`\nPairs: ${rows.length} (${tested.length} pass the ${S.minMotionDeg}° gate)\n`);
console.log('protocol / file                    motion°  face px  RMS px  RMS/face %  T/dof  relief share  bend share  relief vs real face');
for (const r of rows) {
  console.log(
    `${r.file.slice(0, 34).padEnd(34)} ${r.motion.toFixed(1).padStart(6)} ${r.facePx.toFixed(0).padStart(8)} ${r.rmsPx.toFixed(2).padStart(7)} ${(100 * r.rmsPx / r.facePx).toFixed(2).padStart(11)} ${r.tOverDof.toFixed(0).padStart(6)} ${r.relief.toFixed(2).padStart(13)} ${r.bend.toFixed(2).padStart(11)} ${Number.isFinite(r.kappa) ? (100 * r.kappa).toFixed(0).padStart(17) + ' %' : '                 —'}`,
  );
}

console.log('\n1. Does the error grow with rotation?');
console.log(`   Spearman correlation (rotation, RMS/face size), all ${rows.length} pairs: ${spearman(rows.map((r) => r.motion), rows.map((r) => r.rmsPx / r.facePx)).toFixed(2)}`);
const bins: [number, number][] = [[0, 2], [2, 5], [5, 8], [8, 90]];
for (const [lo, hi] of bins) {
  const b = rows.filter((r) => r.motion >= lo && r.motion < hi);
  if (b.length) console.log(`   rotation ${lo}–${hi}°: ${String(b.length).padStart(2)} pairs, median RMS ${med(b.map((r) => r.rmsPx)).toFixed(2)} px, median relief share ${med(b.map((r) => r.relief)).toFixed(2)}`);
}

console.log('\n2. Is the pattern face relief or a bent sheet? (tested pairs)');
console.log(`   median share explained by face relief: ${med(tested.map((r) => r.relief)).toFixed(2)}; by a vertical bend: ${med(tested.map((r) => r.bend)).toFixed(2)}`);
console.log(`   relief wins in ${tested.filter((r) => r.relief > r.bend).length} of ${tested.length} pairs; correlation of the two patterns: ${(() => {
  let a = 0, b = 0, c = 0;
  relief.forEach((d, i) => { a += d * bend[i]; b += d * d; c += bend[i] * bend[i]; });
  return (a / Math.sqrt(b * c)).toFixed(2);
})()}`);
console.log(`   relief signal compared with a real 3D face at the same angle and size: median ${(100 * med(tested.map((r) => r.kappa).filter(Number.isFinite))).toFixed(0)} %`);

console.log('\n3. Which landmarks carry the error? (tested pairs; mean |residual| / pair RMS)');
const per = idx.map(() => 0);
for (const r of tested) {
  const rms = Math.sqrt(r.res.reduce((s, v) => s + v.x * v.x + v.y * v.y, 0) / r.res.length);
  r.res.forEach((v, i) => (per[i] += Math.hypot(v.x, v.y) / rms / tested.length));
}
const order = idx.map((li, k) => ({ li, k })).sort((a, b) => per[b.k] - per[a.k]);
for (const { li, k } of order) {
  console.log(`   landmark ${String(li).padStart(3)}: ${per[k].toFixed(2)}   height above face plane ${relief[k] >= 0 ? '+' : ''}${relief[k].toFixed(1)} cm`);
}
console.log(`   Spearman (|height above plane|, error): ${spearman(relief.map(Math.abs), per).toFixed(2)}`);

console.log('\n4. How large would σ have to be for the χ² test to accept the print? (tested pairs)');
for (const s of [0.1, 0.25, 0.5, 1, 1.5, 2, 3]) {
  const rej = tested.filter((r) => r.C / (s * s) > CHI2_95_42).length;
  console.log(`   σ = ${String(s).padEnd(4)} px → rejected ${rej}/${tested.length}`);
}

console.log('\nReference (synthetic, σ = 0.5 px, 7° yaw, 200 runs): median relief share');
const rng = new Rng(1);
for (const kind of ['plane', 'face3d'] as const) {
  const sh: number[] = [];
  for (let k = 0; k < 200; k++) {
    const c = syntheticPair({ object: makeObject(kind), pose0: { yawDeg: -3.5, pitchDeg: 0 }, pose1: { yawDeg: 3.5, pitchDeg: 0 }, distanceCm: 45, sigmaPx: 0.5, indices: idx }, rng);
    sh.push(patternFit(analyzePair(c, { sigmaPx: 0.5, alpha: 0.05, minMotionDeg: 0, motionDeg: 7 }).residualVectors!, relief).share);
  }
  console.log(`   ${kind.padEnd(6)} ${med(sh).toFixed(2)}`);
}
