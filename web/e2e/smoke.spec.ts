import { execSync, spawn } from 'node:child_process';
import { mkdtempSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test, type ConsoleMessage } from '@playwright/test';

// Most checks run in English; the language test switches explicitly.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    try {
      if (!sessionStorage.getItem('e2e-lang-set')) {
        localStorage.setItem('parallax-lab-lang', 'en');
        sessionStorage.setItem('e2e-lang-set', '1');
      }
    } catch {
      /* ignore */
    }
  });
});

function collectErrors(page: import('@playwright/test').Page): string[] {
  const errors: string[] = [];
  page.on('console', (m: ConsoleMessage) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

test('every section renders without errors and declares the CSP', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  const csp = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute('content');
  expect(csp).toContain("connect-src 'self'");
  for (const [hash, heading] of [
    ['live', 'Live planarity test'],
    ['experiment', 'Experiment: is a moving flat print planar to MediaPipe?'],
    ['record', 'Record'],
    ['analyze', 'Analyze recordings'],
    ['synthetic', 'Synthetic lab'],
    ['diagnostics', 'Diagnostics'],
    ['method', 'Method and status'],
    ['about', 'About'],
  ]) {
    await page.goto(`/#${hash}`);
    await expect(page.getByRole('heading', { level: 1, name: heading })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('method page reports all six method functions as implemented', async ({ page }) => {
  await page.goto('/#method');
  await expect(page.getByText('6 of 6 method functions are implemented')).toBeVisible();
  await expect(page.getByText('None yet.')).toBeVisible();
});

test('synthetic sweep runs and fills the results table', async ({ page }) => {
  await page.goto('/#synthetic');
  await page.getByRole('button', { name: 'Run sweep' }).click();
  await expect(page.getByText(/Done in .* s\. Seed 2024/)).toBeVisible({ timeout: 60_000 });
  // 3 objects x 7 rotations, plus the header row.
  await expect(page.locator('.data-table tbody tr')).toHaveCount(21);
});

test('camera + MediaPipe run under the CSP (fake camera, no face)', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/#live');
  await page.locator('main').getByRole('combobox').nth(1).selectOption('CPU');
  await page.getByRole('button', { name: 'Start camera' }).click();
  await expect(page.getByRole('button', { name: 'Stop' })).toBeEnabled({ timeout: 60_000 });
  // The "starting…" overlay must disappear once the camera runs.
  await expect(page.locator('.stage-placeholder')).toBeHidden();
  // Frames are processed: the frame-rate measurement appears.
  await expect(page.locator('.kv', { hasText: 'Frame rate' }).locator('.kv-value')).toHaveText(/fps/, { timeout: 30_000 });
  await page.goto('/#diagnostics');
  const processed = page.locator('.kv', { hasText: 'Frames processed' }).locator('.kv-value');
  await expect.poll(async () => Number(await processed.textContent()), { timeout: 20_000 }).toBeGreaterThan(10);
  const detection = await page.locator('.kv', { hasText: 'Face detection rate' }).locator('.kv-value').textContent();
  const inference = await page.locator('.kv', { hasText: 'Inference mean' }).locator('.kv-value').textContent();
  console.log(`fake camera: ${await processed.textContent()} frames, detection rate ${detection}, inference mean/p95 ${inference}`);
  // No CSP violations or runtime errors. MediaPipe prints informational
  // lines ("INFO: Created TensorFlow Lite XNNPACK delegate…") via console.error.
  expect(errors.filter((e) => !e.startsWith('INFO:'))).toEqual([]);
});

test('record view blocks recording people without an ethics approval reference', async ({ page }) => {
  await page.goto('/#record');
  // Person-only fields stay hidden while a print is selected.
  await expect(page.getByPlaceholder('P01')).toBeHidden();
  await page.getByRole('button', { name: 'Start camera' }).click();
  const recordBtn = page.getByRole('button', { name: 'Record', exact: true });
  await expect(recordBtn).toBeEnabled({ timeout: 60_000 });
  await page.locator('main').getByRole('combobox').first().selectOption('human-participant');
  await expect(recordBtn).toBeDisabled();
  await expect(page.getByText('Recording a person requires an ethics approval reference.')).toBeVisible();
  await page.getByPlaceholder(/ethics committee/).fill('TEST-APPROVAL-1');
  await page.getByPlaceholder('P01').fill('P01');
  await expect(recordBtn).toBeDisabled();
  await page.getByRole('checkbox').check();
  await expect(recordBtn).toBeEnabled();
});

test('interface switches to Russian and Kazakh without leftover placeholders', async ({ page }) => {
  await page.goto('/#live');
  await page.getByRole('combobox', { name: 'Language' }).selectOption('ru');
  await expect(page.getByRole('heading', { level: 1, name: 'Живой тест плоскостности' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Эксперимент' })).toBeVisible();
  for (const hash of ['experiment', 'record', 'analyze', 'synthetic', 'diagnostics', 'method', 'about']) {
    await page.goto(`/#${hash}`);
    const text = await page.locator('main').innerText();
    expect(text, hash).not.toMatch(/\{[a-zA-Z]+\}/);
  }
  await page.getByRole('combobox', { name: 'Язык' }).selectOption('kk');
  await expect(page.getByRole('heading', { level: 1, name: 'Жоба туралы' })).toBeVisible();
  await expect(page.getByText(/тіл иесі тексерген жоқ/)).toBeVisible();
  // The choice persists across reloads.
  await page.reload();
  await expect(page.getByRole('heading', { level: 1, name: 'Жоба туралы' })).toBeVisible();
});

test('analyze view loads synthetic recordings and reports results by condition', async ({ page }) => {
  const dir = mkdtempSync(join(tmpdir(), 'parallax-e2e-'));
  execSync(`npx tsx scripts/make-synthetic-recordings.ts ${dir}`, { stdio: 'ignore' });
  const files = readdirSync(dir).map((f) => join(dir, f));
  await page.goto('/#analyze');
  await page.locator('input[type=file]').setInputFiles(files);
  await expect(page.locator('.data-table tbody tr')).toHaveCount(4);
  await page.getByRole('button', { name: 'Analyze', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Results by condition' })).toBeVisible();
  for (const c of ['synthetic-plane', 'synthetic-face3d', 'synthetic-cylinder']) {
    await expect(page.locator('.data-table td', { hasText: c }).first()).toBeVisible();
  }
});

test('experiment wizard requires confirmation and a running camera, and reports a failed calibration', async ({ page }) => {
  await page.goto('/#experiment');
  const next = page.getByRole('button', { name: 'Next' });
  await expect(next).toBeDisabled();
  await page.getByRole('checkbox').check();
  await expect(next).toBeDisabled(); // camera not running yet
  await page.getByRole('button', { name: 'Start camera' }).click();
  await expect(next).toBeEnabled({ timeout: 60_000 });
  await next.click();
  await page.getByRole('button', { name: 'Record calibration' }).click();
  // The fake camera shows no face, so nothing can be recorded.
  await expect(page.getByText('Nothing recorded: no face was found in the frames.')).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole('button', { name: 'Next' })).toBeDisabled();
});

// The app is served by a separate server that the test kills half-way, so
// "offline" is real. (Playwright's context.setOffline does not apply to
// requests made by a service worker in Chromium, so it cannot be used here.)
test('after the first visit the app and the face model work offline', async ({ page }) => {
  const errors = collectErrors(page);
  const port = 4175;
  const server = spawn('npx', ['vite', 'preview', '--port', String(port), '--strictPort'], { detached: true, stdio: 'ignore' });
  const killServer = () => {
    try {
      process.kill(-server.pid!, 'SIGTERM');
    } catch {
      /* already gone */
    }
  };
  try {
    const base = `http://localhost:${port}/`;
    await expect.poll(() => fetch(base).then((r) => r.status, () => 0), { timeout: 30_000 }).toBe(200);
    await page.goto(`${base}#diagnostics`);
    await expect(page.locator('.kv', { hasText: 'Works offline' })).toBeVisible();
    // The worker claims the page once its precache is complete.
    await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller), { timeout: 30_000 }).toBe(true);

    killServer();
    await expect.poll(() => fetch(base).then(() => 'up', () => 'down'), { timeout: 10_000 }).toBe('down');
    const failed: string[] = [];
    page.on('requestfailed', (r) => failed.push(new URL(r.url()).pathname));

    await page.reload();
    await expect(page.getByRole('heading', { level: 1, name: 'Diagnostics' })).toBeVisible();
    await expect(page.locator('.kv', { hasText: 'Works offline' })).toContainText('yes');
    // Control: a file that is not precached cannot load, so the server really is gone.
    const uncached = await page.evaluate(() =>
      fetch('wasm/vision_wasm_nosimd_internal.js').then(
        () => 'loaded',
        () => 'failed',
      ),
    );
    expect(uncached).toBe('failed');
    // The face model loads and runs without the server (the fake camera shows no face).
    await page.goto(`${base}#live`);
    await page.locator('main').getByRole('combobox').nth(1).selectOption('CPU');
    await page.getByRole('button', { name: 'Start camera' }).click();
    await expect(page.getByRole('button', { name: 'Stop' })).toBeEnabled({ timeout: 60_000 });
    await expect(page.locator('.kv', { hasText: 'Frame rate' }).locator('.kv-value')).toHaveText(/fps/, { timeout: 30_000 });
    // Only the control request may fail (the browser logs it as a console error).
    expect(failed).toEqual(['/wasm/vision_wasm_nosimd_internal.js']);
    expect(errors.filter((e) => !e.startsWith('INFO:') && e !== 'Failed to load resource: net::ERR_FAILED')).toEqual([]);
  } finally {
    killServer();
  }
});

test('number fields revert an empty entry and clamp out-of-range values', async ({ page }) => {
  await page.goto('/#analyze');
  const alpha = page.locator('main label', { hasText: 'α' }).locator('input[type=number]');
  await alpha.fill('');
  await alpha.dispatchEvent('change');
  await expect(alpha).toHaveValue('0.05');
  await alpha.fill('3');
  await alpha.dispatchEvent('change');
  await expect(alpha).toHaveValue('0.5');
});
