// End-to-end UI tests.
//
// These drive the real workbench in a headless browser and assert on what the
// page actually renders — the numbers come from the map, not from a mock.

import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';

import {
  startServer, launchBrowser, waitForSolves, currentSolves,
} from '../helpers/browser.js';

let server;
let browser;
let page;
let skipReason = null;

before(async () => {
  server = await startServer();
  browser = await launchBrowser();
  if (!browser) {
    skipReason = 'no Chromium binary available for E2E';
    return;
  }
  page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on('pageerror', (e) => { skipReason = skipReason ?? `pageerror: ${e.message}`; });
  await page.goto(`${server.url}/`, { waitUntil: 'load' });
  await waitForSolves(page, 0);
});

after(async () => {
  if (browser) await browser.close();
  if (server) server.stop();
});

function requireBrowser(t) {
  if (!browser) { t.skip(skipReason ?? 'no browser'); return false; }
  return true;
}

test('synthetic dataset solves on load and renders a safe window', async (t) => {
  if (!requireBrowser(t)) return;
  const status = await page.locator('#status').innerText();
  assert.match(status, /solved/i);
  const windowText = await page.locator('#window-out').innerText();
  assert.match(windowText, /decision/);
  assert.match(windowText, /accepted|shrunk/);
  // The window must be reported from the actual map samples.
  assert.match(windowText, /map samples/);
  assert.doesNotMatch(windowText, /NaN|Infinity/);
});

test('switching to the per-focal path re-solves', async (t) => {
  if (!requireBrowser(t)) return;
  const before = await currentSolves(page);
  await page.selectOption('#path-select', 'per');
  await waitForSolves(page, before);
  const fit = await page.locator('#fit-out').innerText();
  assert.match(fit, /per|shared/);
});

test('deleting the tilted wide view changes the per-focal safe crop', async (t) => {
  if (!requireBrowser(t)) return;
  await page.selectOption('#view-select', 'wide#0');

  // Per-focal baseline (the tilted wide#4 view included).
  let n = await currentSolves(page);
  await page.selectOption('#path-select', 'per');
  n = await waitForSolves(page, n);
  const keep = await page.locator('#window-out').innerText();
  const keepPx = parseWindow(keep);

  // Delete wide#4 — the only wide view that reaches the sensor corners.
  n = await currentSolves(page);
  await page.locator('.view-item', { hasText: 'wide#4' })
    .locator('button', { hasText: /delete/i }).click();
  n = await waitForSolves(page, n);
  const del = await page.locator('#window-out').innerText();
  const delPx = parseWindow(del);

  assert.ok(keepPx && delPx, 'both windows parsed');
  const shrunk = delPx.width < keepPx.width || delPx.height < keepPx.height;
  assert.ok(shrunk, `safe crop must change when the only peripheral constraint is removed (kept ${fmt(keepPx)} -> ${fmt(delPx)})`);

  // Restore for later tests.
  n = await currentSolves(page);
  await page.locator('.view-item', { hasText: 'wide#4' })
    .locator('button', { hasText: /restore/i }).click();
  await waitForSolves(page, n);
});

test('the shared path stays anchored when the same view is deleted', async (t) => {
  if (!requireBrowser(t)) return;
  await page.selectOption('#view-select', 'wide#0');
  let n = await currentSolves(page);
  await page.selectOption('#path-select', 'shared');
  n = await waitForSolves(page, n);
  const keep = parseWindow(await page.locator('#window-out').innerText());

  n = await currentSolves(page);
  await page.locator('.view-item', { hasText: 'wide#4' })
    .locator('button', { hasText: /delete/i }).click();
  n = await waitForSolves(page, n);
  const del = parseWindow(await page.locator('#window-out').innerText());

  // The other segments anchor the shared distortion, so the crop must not
  // collapse the way the per-focal path does.
  const areaKeep = keep.width * keep.height;
  const areaDel = del.width * del.height;
  assert.ok(areaDel > 0.85 * areaKeep,
    `shared crop stays anchored (${fmt(keep)} -> ${fmt(del)})`);

  n = await currentSolves(page);
  await page.locator('.view-item', { hasText: 'wide#4' })
    .locator('button', { hasText: /restore/i }).click();
  await waitForSolves(page, n);
});

test('the undistorted view actually resamples (no blank output)', async (t) => {
  if (!requireBrowser(t)) return;
  let n = await currentSolves(page);
  await page.selectOption('#canvas-mode', 'undistorted');
  await page.waitForTimeout(400);
  const blank = await page.evaluate(() => {
    const canvas = document.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let transparent = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i] === 0) transparent++;
    return transparent / (data.length / 4);
  });
  assert.ok(blank < 0.5, `most of the undistorted frame is opaque (blank fraction ${blank.toFixed(3)})`);
  await page.selectOption('#canvas-mode', 'raw');
});

test('the real-photo dataset solves with honest held-out metrics', async (t) => {
  if (!requireBrowser(t)) return;
  const n = await currentSolves(page);
  await page.locator('input[name="dataset"][value="real"]').check();
  await waitForSolves(page, n);
  const verify = await page.locator('#verify-out').innerText();
  assert.match(verify, /held-out corner/);
  assert.match(verify, /lattice bend/);
  assert.doesNotMatch(verify, /NaN|Infinity/);
  // The detector should recover the 9x6 board on every real view.
  const summary = await page.locator('#views-summary').innerText();
  assert.match(summary, /13\/13/);
  const residuals = await page.locator('#residuals-out').innerText();
  // No per-view residual should be absurd (a scrambled board would be >1 px).
  const values = [...residuals.matchAll(/([\d.]+)\s*px/g)].map((m) => Number(m[1]));
  assert.ok(values.length > 0);
  for (const v of values) assert.ok(v < 2, `per-view residual ${v}px stays sub-2px`);
});

function parseWindow(text) {
  const m = text.match(/window px\s*\n?\s*(\d+)\s*×\s*(\d+)/);
  if (!m) return null;
  const o = text.match(/origin px\s*\n?\s*(-?\d+),\s*(-?\d+)/);
  return {
    width: Number(m[1]), height: Number(m[2]),
    x0: o ? Number(o[1]) : 0, y0: o ? Number(o[2]) : 0,
  };
}

function fmt(w) { return `${w.width}x${w.height}@${w.x0},${w.y0}`; }
