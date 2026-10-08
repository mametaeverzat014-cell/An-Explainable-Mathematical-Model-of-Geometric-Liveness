import { defineConfig, type Plugin } from 'vite';
import { studentAlias } from './studentAlias.ts';

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

export default defineConfig({
  base: './',
  resolve: { alias: studentAlias() },
  plugins: [injectCsp()],
  build: { target: 'es2022', sourcemap: true },
});
