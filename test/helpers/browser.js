// Browser fixture for the E2E tests.
//
// Uses playwright-core with the Chromium headless_shell that is already
// installed in this environment (the full chromium bundle is missing shared
// libraries). Tests are skipped gracefully when no browser is available so the
// unit suite stays runnable anywhere.

import os from 'node:os';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

const CANDIDATES = [
  path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1148/chrome-linux/headless_shell'),
  path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1148/chrome-linux/chrome'),
  process.env.CHROME_PATH,
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/usr/bin/google-chrome',
].filter(Boolean);

function statSyncSafe(p) {
  try { return fs.statSync(p).isFile(); } catch { return false; }
}

export function findBrowser() {
  for (const p of CANDIDATES) {
    if (statSyncSafe(p)) return p;
  }
  return null;
}

// Start the static server on a free-ish port and resolve when it answers.
export async function startServer(port = 0) {
  const env = { ...process.env };
  if (!env.LD_LIBRARY_PATH) {
    env.LD_LIBRARY_PATH = '/tmp/debs/root/lib/aarch64-linux-gnu:/tmp/debs/root/usr/lib/aarch64-linux-gnu';
  }
  const chosen = port || 4300 + Math.floor(Math.random() * 500);
  const child = spawn(process.execPath, [path.join(ROOT, 'tools/serve.js')], {
    cwd: ROOT,
    env: { ...env, PORT: String(chosen), HOST: '127.0.0.1' },
    stdio: 'ignore',
  });
  const url = `http://127.0.0.1:${chosen}`;
  await waitForServer(url, 5000);
  return { url, port: chosen, stop: () => child.kill('SIGTERM') };
}

async function waitForServer(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url}/src/app/index.html`, { method: 'HEAD' });
      if (res.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`server did not start at ${url}`);
}

export async function launchBrowser() {
  const executablePath = findBrowser();
  if (!executablePath) return null;
  const env = { ...process.env };
  if (!env.LD_LIBRARY_PATH) {
    env.LD_LIBRARY_PATH = '/tmp/debs/root/lib/aarch64-linux-gnu:/tmp/debs/root/usr/lib/aarch64-linux-gnu';
  }
  return chromium.launch({
    executablePath,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
    env,
  });
}

// Wait until the app has finished `n` solves (the status node counts them).
export async function waitForSolves(page, previous, timeout = 120000) {
  await page.waitForFunction(
    (p) => {
      const el = document.getElementById('status');
      return el && el.dataset.solves && Number(el.dataset.solves) > p;
    },
    previous,
    { timeout },
  );
  return Number(await page.locator('#status').getAttribute('data-solves'));
}

export async function currentSolves(page) {
  return Number(await page.locator('#status').getAttribute('data-solves')) || 0;
}
