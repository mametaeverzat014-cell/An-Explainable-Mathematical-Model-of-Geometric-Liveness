// One-off provenance script: converts MediaPipe's canonical_face_model.obj
// (Apache-2.0, google-ai-edge/mediapipe,
// mediapipe/modules/face_geometry/data/canonical_face_model.obj) into the
// JSON vertex list used by the synthetic scene. Units are centimetres; axes:
// x to the subject's left on the image, y up, z towards the viewer.
//
// Usage: node scripts/convert-canonical-face.mjs path/to/canonical_face_model.obj
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const objPath = process.argv[2];
if (!objPath) {
  console.error('usage: node scripts/convert-canonical-face.mjs <canonical_face_model.obj>');
  process.exit(1);
}
const text = readFileSync(objPath, 'utf8');
const vertices = text
  .split('\n')
  .filter((line) => line.startsWith('v '))
  .map((line) => line.trim().split(/\s+/).slice(1, 4).map(Number));
if (vertices.length !== 468) throw new Error(`expected 468 vertices, got ${vertices.length}`);
const sha256 = createHash('sha256').update(text).digest('hex');
const out = {
  source: 'google-ai-edge/mediapipe: mediapipe/modules/face_geometry/data/canonical_face_model.obj',
  license: 'Apache-2.0',
  sourceSha256: sha256,
  units: 'cm',
  vertices,
};
writeFileSync(new URL('../src/data/canonical-face.json', import.meta.url), JSON.stringify(out) + '\n');
console.log(`wrote ${vertices.length} vertices (source sha256 ${sha256})`);
