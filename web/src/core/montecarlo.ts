import { analyzePair, DEFAULT_METHOD, type PlanarityMethod } from './method';
import { Rng } from './rng';
import { wilsonInterval } from './stats';
import { makeObject, syntheticPair, type Intrinsics, type ObjectKind } from './synthetic';

// Monte-Carlo estimation of the test's rejection rate on the synthetic scene.
// For a planar object the rejection rate estimates the test SIZE (false
// "non-planar" rate); for the 3D face it estimates POWER. Results are
// conditional on the synthetic assumptions (pinhole camera, i.i.d. Gaussian
// noise of KNOWN sigma) and are not measurements of real-world performance.

export interface SweepSpec {
  kinds: ObjectKind[];
  /** Rotation magnitudes to evaluate, degrees. */
  rotationsDeg: number[];
  axis: 'yaw' | 'pitch';
  sigmaPx: number;
  distanceCm: number;
  indices: readonly number[];
  trials: number;
  alpha: number;
  seed: number;
  intrinsics?: Intrinsics;
}

export interface SweepPoint {
  kind: ObjectKind;
  rotationDeg: number;
  trials: number;
  rejections: number;
  rate: number;
  ci95: [number, number];
  meanTOverDof: number;
}

export interface SweepProgress {
  done: number;
  total: number;
}

export type SweepResult = { ok: true; points: SweepPoint[] } | { ok: false; reason: string };

/**
 * Run the sweep, yielding to the event loop between chunks so the UI stays
 * responsive. The motion gate is disabled here (minMotionDeg = 0) because
 * the sweep measures the statistical test itself.
 */
export async function runSweep(
  spec: SweepSpec,
  onProgress?: (p: SweepProgress) => void,
  method: PlanarityMethod = DEFAULT_METHOD,
  shouldStop: () => boolean = () => false,
): Promise<SweepResult> {
  const total = spec.kinds.length * spec.rotationsDeg.length * spec.trials;
  let done = 0;
  const points: SweepPoint[] = [];
  for (const [ki, kind] of spec.kinds.entries()) {
    const object = makeObject(kind);
    for (const [ri, rotationDeg] of spec.rotationsDeg.entries()) {
      // Independent, reproducible stream per (kind, rotation) cell.
      const rng = new Rng(spec.seed + 1000 * ki + ri);
      let rejections = 0;
      let sumTOverDof = 0;
      for (let t = 0; t < spec.trials; t++) {
        if (shouldStop()) return { ok: false, reason: 'stopped' };
        const pose1 = spec.axis === 'yaw' ? { yawDeg: rotationDeg, pitchDeg: 0 } : { yawDeg: 0, pitchDeg: rotationDeg };
        const corr = syntheticPair(
          { object, pose0: { yawDeg: 0, pitchDeg: 0 }, pose1, distanceCm: spec.distanceCm, sigmaPx: spec.sigmaPx, indices: spec.indices, intrinsics: spec.intrinsics },
          rng,
        );
        const a = analyzePair(corr, { sigmaPx: spec.sigmaPx, alpha: spec.alpha, minMotionDeg: 0, motionDeg: rotationDeg }, method);
        const blocked = a.stages.find((s) => s.state === 'not-implemented' || s.state === 'error');
        if (blocked) return { ok: false, reason: `${blocked.stage}: ${blocked.message ?? blocked.state}` };
        if (a.stat && a.stat.pValue < spec.alpha) rejections++;
        if (a.stat) sumTOverDof += a.stat.T / a.stat.dof;
        done++;
        if (done % 50 === 0) {
          onProgress?.({ done, total });
          await new Promise((r) => setTimeout(r, 0));
        }
      }
      points.push({
        kind,
        rotationDeg,
        trials: spec.trials,
        rejections,
        rate: rejections / spec.trials,
        ci95: wilsonInterval(rejections, spec.trials),
        meanTOverDof: sumTOverDof / spec.trials,
      });
    }
  }
  onProgress?.({ done: total, total });
  return { ok: true, points };
}
