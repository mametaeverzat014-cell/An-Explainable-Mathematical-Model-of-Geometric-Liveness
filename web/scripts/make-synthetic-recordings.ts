// Writes example recordings generated from the synthetic scene, in the same
// format as real ones, so the analysis path can be tried before real data
// exist. Usage: npm run make-synthetic -- [outDir]
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { syntheticRecording } from '../src/core/recording-analysis';
import { recordingFileName } from '../src/core/recording';

const outDir = process.argv[2] ?? 'recordings/synthetic';
mkdirSync(outDir, { recursive: true });
const common = { periodMs: 4000, durationMs: 12000, fps: 30, distanceCm: 50, sigmaPx: 0.8 };
const specs = [
  { ...common, kind: 'plane' as const, role: 'calibration' as const, yawAmplitudeDeg: 0, pitchAmplitudeDeg: 0, durationMs: 3000, seed: 1 },
  { ...common, kind: 'plane' as const, role: 'trial' as const, yawAmplitudeDeg: 20, pitchAmplitudeDeg: 10, seed: 2 },
  { ...common, kind: 'face3d' as const, role: 'trial' as const, yawAmplitudeDeg: 20, pitchAmplitudeDeg: 10, seed: 3 },
  { ...common, kind: 'cylinder' as const, role: 'trial' as const, yawAmplitudeDeg: 20, pitchAmplitudeDeg: 10, seed: 4 },
];
for (const spec of specs) {
  const rec = syntheticRecording(spec);
  const path = join(outDir, recordingFileName(rec.meta));
  writeFileSync(path, JSON.stringify(rec));
  console.log(`wrote ${path} (${rec.frames.length} frames)`);
}
