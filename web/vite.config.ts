import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import { methodAlias } from './methodAlias.ts';

// Content-Security-Policy injected into the production build only (the dev
// server needs inline scripts and a websocket for hot reload). connect-src
// 'self' means the page cannot send data to any other origin: camera frames
// and landmarks stay on the device. 'wasm-unsafe-eval' is required to
// instantiate the MediaPipe WebAssembly module.
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "connect-src 'self'",
  "img-src 'self' data: blob:",
  "media-src 'self' blob: mediastream:",
  "worker-src 'self' blob:",
  "style-src 'self' 'unsafe-inline'",
  "font-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
].join('; ');

function injectCsp(): Plugin {
  return {
    name: 'inject-csp',
    apply: 'build',
    transformIndexHtml(html) {
      return html.replace(
        '<meta charset="UTF-8" />',
        `<meta charset="UTF-8" />\n    <meta http-equiv="Content-Security-Policy" content="${CSP}" />`,
      );
    },
  };
}

// Files the service worker does NOT download in advance: source maps, and the
// WebAssembly variants that only some browsers use (ES-module build; build
// for browsers without SIMD). They are cached the first time they are used.
const NOT_PRECACHED = [/\.map$/, /^wasm\/vision_wasm_module_internal\./, /^wasm\/vision_wasm_nosimd_internal\./];

function listFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = join(dir, name);
    return statSync(full).isDirectory() ? listFiles(full) : [full];
  });
}

// Writes dist/sw.js from src/offline/service-worker.js with the list of files
// to precache. The version is a hash of those files and of the worker
// itself, so every change to the deployed app produces a new worker and a
// fresh cache.
function offlinePlugin(): Plugin {
  let outDir = '';
  let template = '';
  return {
    name: 'offline-service-worker',
    apply: 'build',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
      template = resolve(config.root, 'src/offline/service-worker.js');
    },
    closeBundle() {
      const source = readFileSync(template, 'utf8');
      const files = listFiles(outDir)
        .map((f) => relative(outDir, f).split(sep).join('/'))
        .filter((f) => f !== 'sw.js' && !NOT_PRECACHED.some((re) => re.test(f)))
        .sort();
      const hash = createHash('sha256').update(source);
      for (const f of files) hash.update(f).update(readFileSync(join(outDir, f)));
      const version = hash.digest('hex').slice(0, 16);
      const sw = source
        .replace('const VERSION = __VERSION__;', `const VERSION = ${JSON.stringify(version)};`)
        .replace('const PRECACHE = __PRECACHE__;', `const PRECACHE = ${JSON.stringify(files, null, 2)};`);
      if (sw.includes('= __VERSION__') || sw.includes('= __PRECACHE__')) throw new Error('service worker template placeholders not replaced');
      writeFileSync(join(outDir, 'sw.js'), sw);
    },
  };
}

export default defineConfig({
  base: './',
  resolve: { alias: methodAlias() },
  plugins: [injectCsp(), offlinePlugin()],
  build: { target: 'es2022', sourcemap: true },
});
