// Shared helpers for node:test suites.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import jpeg from 'jpeg-js';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

export function readJpegAsImageData(relPath) {
  const buf = fs.readFileSync(path.join(ROOT, relPath));
  const raw = jpeg.decode(buf, { useTArray: true });
  return { data: raw.data, width: raw.width, height: raw.height };
}

export function readJson(relPath) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, relPath), 'utf8'));
}

export function listSamples() {
  return fs.readdirSync(path.join(ROOT, 'data/samples'))
    .filter((f) => f.endsWith('.jpg'))
    .sort();
}

// Bijective match from detected points to reference corners. Both sets are the
// same physical corners but may be indexed in a different axis order/mirroring,
// so matching by proximity is the correct comparison.
export function setMatchError(detected, reference, tolPx = 5) {
  const used = new Set();
  let matched = 0;
  let sum = 0;
  let worst = 0;
  for (const p of detected) {
    let best = Infinity, bi = -1;
    for (let i = 0; i < reference.length; i++) {
      if (used.has(i)) continue;
      const d = Math.hypot(p[0] - reference[i][0], p[1] - reference[i][1]);
      if (d < best) { best = d; bi = i; }
    }
    if (bi >= 0 && best < tolPx) {
      used.add(bi);
      matched++;
      sum += best;
      worst = Math.max(worst, best);
    }
  }
  return { matched, total: reference.length, meanErr: matched ? sum / matched : Infinity, worstErr: worst };
}
