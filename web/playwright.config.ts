import { defineConfig } from '@playwright/test';

// End-to-end smoke tests against the PRODUCTION build (so the
// Content-Security-Policy is active). Chromium's fake camera supplies a
// synthetic test pattern; it contains no face, which exercises the
// "no face detected" path and proves the MediaPipe runtime loads under the CSP.
export default defineConfig({
  testDir: 'e2e',
  timeout: 90_000,
  use: {
    baseURL: 'http://localhost:4173',
    launchOptions: {
      executablePath: process.env.CHROMIUM_PATH || undefined,
      args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
    },
  },
  webServer: {
    command: 'npx vite preview --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
