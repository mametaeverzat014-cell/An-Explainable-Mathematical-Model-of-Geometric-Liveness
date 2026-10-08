import { expect, test, type ConsoleMessage } from '@playwright/test';

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
    ['record', 'Record'],
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
  await page.getByRole('combobox').nth(1).selectOption('CPU');
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
  await page.getByRole('combobox').first().selectOption('human-participant');
  await expect(recordBtn).toBeDisabled();
  await expect(page.getByText('Recording a person requires an ethics approval reference.')).toBeVisible();
  await page.getByPlaceholder(/ethics committee/).fill('TEST-APPROVAL-1');
  await page.getByPlaceholder('P01').fill('P01');
  await expect(recordBtn).toBeDisabled();
  await page.getByRole('checkbox').check();
  await expect(recordBtn).toBeEnabled();
});
