import { FaceLandmarker } from '@mediapipe/tasks-vision';
import { describe, expect, it } from 'vitest';
import { FACE_OVAL, LANDMARK_SETS, LEFT_EYE, LEFT_EYEBROW, LIPS, RIGHT_EYE, RIGHT_EYEBROW } from '../landmark-sets';

const fromPackage = (connections: { start: number; end: number }[]) =>
  [...new Set(connections.flatMap((c) => [c.start, c.end]))].sort((a, b) => a - b);

describe('landmark sets', () => {
  it('region lists match the installed @mediapipe/tasks-vision package', () => {
    expect([...FACE_OVAL]).toEqual(fromPackage(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL));
    expect([...LEFT_EYE]).toEqual(fromPackage(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE));
    expect([...RIGHT_EYE]).toEqual(fromPackage(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE));
    expect([...LEFT_EYEBROW]).toEqual(fromPackage(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW));
    expect([...RIGHT_EYEBROW]).toEqual(fromPackage(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW));
    expect([...LIPS]).toEqual(fromPackage(FaceLandmarker.FACE_LANDMARKS_LIPS));
  });

  it('subsets have the documented sizes, are sorted, unique and within 0..467', () => {
    expect(LANDMARK_SETS.rigid.indices).toHaveLength(25);
    expect(LANDMARK_SETS.interior.indices).toHaveLength(468 - 36);
    expect(LANDMARK_SETS.all.indices).toHaveLength(468);
    for (const set of Object.values(LANDMARK_SETS)) {
      const idx = set.indices;
      expect(new Set(idx).size).toBe(idx.length);
      idx.forEach((v, i) => {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(468);
        if (i > 0) expect(v).toBeGreaterThan(idx[i - 1]);
      });
    }
  });

  it('rigid set contains no face-outline points', () => {
    const oval = new Set<number>(FACE_OVAL);
    expect(LANDMARK_SETS.rigid.indices.some((i) => oval.has(i))).toBe(false);
  });
});
