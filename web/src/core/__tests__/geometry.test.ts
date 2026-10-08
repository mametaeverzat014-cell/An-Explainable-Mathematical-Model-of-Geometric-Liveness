import { describe, expect, it } from 'vitest';
import { relativeRotationDeg, rotApply, rotationFrom4x4, rotationFromAngles } from '../geometry3d';
import { LANDMARK_SETS } from '../landmark-sets';
import { CANONICAL_FACE, DEFAULT_INTRINSICS, addNoise, makeObject, projectObject, syntheticPair } from '../synthetic';
import { Rng } from '../rng';
import { mean, sampleVariance } from '../stats';
import { symmetricEigen } from '../linalg';

describe('rotations', () => {
  it('relative rotation angle recovers a single-axis rotation', () => {
    expect(relativeRotationDeg(rotationFromAngles(0, 0), rotationFromAngles(17, 0))).toBeCloseTo(17, 9);
    expect(relativeRotationDeg(rotationFromAngles(10, 0), rotationFromAngles(10, -8))).toBeCloseTo(8, 9);
  });

  it('relative rotation angle is the same for row- and column-major storage', () => {
    const a = rotationFromAngles(12, -7, 3);
    const b = rotationFromAngles(-4, 9, 1);
    const t = (r: number[]) => [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
    expect(relativeRotationDeg(t(a), t(b))).toBeCloseTo(relativeRotationDeg(a, b), 9);
  });

  it('rotationFrom4x4 strips uniform scale', () => {
    const r = rotationFromAngles(20, 5);
    const s = 3;
    const m4 = [s * r[0], s * r[1], s * r[2], 0, s * r[3], s * r[4], s * r[5], 0, s * r[6], s * r[7], s * r[8], 0, 1, 2, 3, 1];
    const back = rotationFrom4x4(m4)!;
    back.forEach((v, i) => expect(v).toBeCloseTo(r[i], 12));
  });

  it('rotations preserve distances', () => {
    const r = rotationFromAngles(33, -21, 8);
    const p = { x: 1, y: 2, z: 3 };
    const q = rotApply(r, p);
    expect(Math.hypot(q.x, q.y, q.z)).toBeCloseTo(Math.hypot(1, 2, 3), 12);
  });
});

describe('canonical face and synthetic scene', () => {
  it('has 468 vertices in centimetres with the expected anatomy', () => {
    expect(CANONICAL_FACE).toHaveLength(468);
    // Nose tip region (index 4) is the most forward point; outer eye corners are ~8.9 cm apart.
    const zMax = Math.max(...CANONICAL_FACE.map((p) => p.z));
    expect(CANONICAL_FACE[4].z).toBe(zMax);
    expect(Math.abs(CANONICAL_FACE[33].x - CANONICAL_FACE[263].x)).toBeCloseTo(8.89, 1);
  });

  it('plane object is exactly planar and cylinder object is not', () => {
    const smallestEig = (kind: 'plane' | 'cylinder') => {
      const pts = makeObject(kind).points;
      const m = [mean(pts.map((p) => p.x)), mean(pts.map((p) => p.y)), mean(pts.map((p) => p.z))];
      const c = [0, 1, 2].map((i) =>
        [0, 1, 2].map((j) => mean(pts.map((p) => ([p.x, p.y, p.z][i] - m[i]) * ([p.x, p.y, p.z][j] - m[j])))),
      );
      return symmetricEigen(c).values[0];
    };
    expect(smallestEig('plane')).toBeLessThan(1e-20);
    expect(smallestEig('cylinder')).toBeGreaterThan(0.1);
  });

  it('projects the frontal face to the image centre region at plausible size', () => {
    const img = projectObject(makeObject('face3d'), { yawDeg: 0, pitchDeg: 0 }, 50);
    const iod = Math.hypot(img[33].x - img[263].x, img[33].y - img[263].y);
    // f * 8.9 cm / depth of eye corners (50 - 3.17 - 4 cm): ~ 203 px
    expect(iod).toBeGreaterThan(180);
    expect(iod).toBeLessThan(230);
    expect(img[1].x).toBeCloseTo(DEFAULT_INTRINSICS.cx, 6);
    // y grows downwards: the forehead (151) is above the chin (152).
    expect(img[151].y).toBeLessThan(img[152].y);
  });

  it('positive yaw moves the nose tip relative to the eye midpoint', () => {
    const face = makeObject('face3d');
    const a = projectObject(face, { yawDeg: 0, pitchDeg: 0 }, 50);
    const b = projectObject(face, { yawDeg: 10, pitchDeg: 0 }, 50);
    const offset = (p: typeof a) => p[1].x - (p[33].x + p[263].x) / 2;
    expect(Math.abs(offset(b) - offset(a))).toBeGreaterThan(5);
  });

  it('noise generator has the requested standard deviation', () => {
    const rng = new Rng(5);
    const pts = Array.from({ length: 20000 }, () => ({ x: 0, y: 0 }));
    const noisy = addNoise(pts, 0.7, rng);
    expect(Math.sqrt(sampleVariance(noisy.map((p) => p.x)))).toBeCloseTo(0.7, 2);
    expect(Math.abs(mean(noisy.map((p) => p.y)))).toBeLessThan(0.02);
  });

  it('syntheticPair is reproducible for a fixed seed', () => {
    const spec = {
      object: makeObject('face3d'),
      pose0: { yawDeg: 0, pitchDeg: 0 },
      pose1: { yawDeg: 5, pitchDeg: 0 },
      distanceCm: 50,
      sigmaPx: 1,
      indices: LANDMARK_SETS.rigid.indices,
    };
    expect(syntheticPair(spec, new Rng(9))).toEqual(syntheticPair(spec, new Rng(9)));
  });
});
