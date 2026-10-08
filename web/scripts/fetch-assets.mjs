// Downloads / copies the third-party runtime assets into public/ so that the
// application loads everything from its own origin (required by the
// Content-Security-Policy, and so that no request leaves the device at run time).
//
// - MediaPipe Face Landmarker model, verified against a pinned SHA-256.
// - MediaPipe Tasks Vision WebAssembly files, copied from node_modules
//   (version pinned in package.json / package-lock.json).
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const MODEL = {
  url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  sha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
  dest: join(root, 'public/models/face_landmarker.task'),
};

function sha256(buf) {
  return createHash('sha256').update(buf).digest('hex');
}

async function fetchModel() {
  if (existsSync(MODEL.dest) && sha256(readFileSync(MODEL.dest)) === MODEL.sha256) {
    console.log('model: present, checksum ok');
    return;
  }
  console.log(`model: downloading ${MODEL.url}`);
  const res = await fetch(MODEL.url);
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const got = sha256(buf);
  if (got !== MODEL.sha256) {
    throw new Error(`model checksum mismatch: expected ${MODEL.sha256}, got ${got}. Refusing to use an unverified model.`);
  }
  mkdirSync(dirname(MODEL.dest), { recursive: true });
  writeFileSync(MODEL.dest, buf);
  console.log('model: downloaded, checksum ok');
}

function copyWasm() {
  const src = join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
  const dest = join(root, 'public/wasm');
  mkdirSync(dest, { recursive: true });
  for (const f of readdirSync(src)) copyFileSync(join(src, f), join(dest, f));
  const version = JSON.parse(readFileSync(join(root, 'node_modules/@mediapipe/tasks-vision/package.json'), 'utf8')).version;
  console.log(`wasm: copied from @mediapipe/tasks-vision ${version}`);
}

await fetchModel();
copyWasm();
