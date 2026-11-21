import { test } from 'node:test';
import assert from 'node:assert/strict';
import zlib from 'node:zlib';

import {
  resampleImage, mapToJSON, sourceAt, encodePng, encodePngRGBA,
} from '../../src/core/resample.js';

// A destination->source identity map with a configurable origin.
function identityMap(w, h, { step = 1, originX = 0, originY = 0 } = {}) {
  const W = Math.ceil(w / step), H = Math.ceil(h / step);
  const mapX = new Float32Array(W * H);
  const mapY = new Float32Array(W * H);
  const valid = new Uint8Array(W * H).fill(1);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      mapX[j * W + i] = originX + i * step;
      mapY[j * W + i] = originY + j * step;
    }
  }
  return { width: W, height: H, step, originX, originY, mapX, mapY, valid,
    stats: { total: W * H, valid: W * H, folds: 0, outside: 0, minJacobianDet: 1 } };
}

function rgba(w, h, fn) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [r, g, b, a] = fn(x, y);
    const o = (y * w + x) * 4;
    data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = a;
  }
  return { data, width: w, height: h };
}

test('resampleImage with an identity map reproduces the source', () => {
  const src = rgba(16, 12, (x, y) => [x * 8, y * 8, (x ^ y) & 255, 255]);
  const map = identityMap(16, 12);
  const out = resampleImage(src, map, { x0: 0, y0: 0, width: 16, height: 12 });
  assert.equal(out.width, 16); assert.equal(out.height, 12);
  assert.equal(out.blank, 0);
  for (let i = 0; i < src.data.length; i++) assert.equal(out.data[i], src.data[i]);
});

test('resampleImage honours a non-zero map origin and output window', () => {
  // Source is a big image; the map only covers a sub-rectangle at (40, 30).
  const src = rgba(100, 80, (x, y) => [x * 2, y * 3, 0, 255]);
  const x0 = 40, y0 = 30, w = 16, h = 12;
  const map = identityMap(w, h, { originX: x0, originY: y0 });
  const out = resampleImage(src, map, { x0, y0, width: w, height: h });
  assert.equal(out.x0, x0); assert.equal(out.y0, y0);
  // The output pixel (0,0) must come from source (40,30), not (0,0).
  assert.equal(out.data[0], 40 * 2);
  assert.equal(out.data[1], 30 * 3);
});

test('resampleImage leaves invalid map samples transparent', () => {
  const src = rgba(8, 8, () => [200, 100, 50, 255]);
  const map = identityMap(8, 8);
  map.valid[0] = 0; // first destination pixel is unusable
  const out = resampleImage(src, map, { x0: 0, y0: 0, width: 8, height: 8 });
  assert.equal(out.data[3], 0, 'invalid sample has zero alpha');
  assert.equal(out.blank, 1);
  assert.equal(out.data[7], 255, 'a valid neighbour keeps full alpha');
});

test('resampleImage blanks destination pixels whose source leaves the frame', () => {
  const src = rgba(8, 8, () => [10, 20, 30, 255]);
  const map = identityMap(8, 8);
  // point one sample outside the source
  map.mapX[0] = -5;
  const out = resampleImage(src, map, { x0: 0, y0: 0, width: 8, height: 8 });
  assert.equal(out.data[3], 0);
});

test('mapToJSON round-trips through sourceAt', () => {
  const map = identityMap(16, 12, { originX: 4, originY: 2 });
  const json = mapToJSON(map, { note: 'test' });
  assert.equal(json.format, 'lens-undistort-map/1');
  assert.equal(json.note, 'test');
  assert.equal(json.mapX.length, map.width * map.height);
  assert.equal(json.valid.length, map.width * map.height);
  // sourceAt must agree with the same nearest-sample rule.
  const s = sourceAt(map, 8, 6);
  assert.ok(s && s.valid);
  assert.equal(s.index, (6 - 2) * map.width + (8 - 4));
  assert.ok(Math.abs(s.u - json.mapX[s.index]) <= 1e-3);
  // Out of range returns null.
  assert.equal(sourceAt(map, -1, 0), null);
  assert.equal(sourceAt(map, 1e6, 1e6), null);
});

test('encodePngRGBA emits a valid PNG signature and IHDR', () => {
  const img = rgba(4, 3, (x, y) => [x * 60, y * 80, 128, 255]);
  const bytes = encodePngRGBA(img.data, img.width, img.height);
  assert.equal(bytes[0], 0x89); assert.equal(bytes[1], 0x50);
  assert.equal(bytes[2], 0x4e); assert.equal(bytes[3], 0x47);
  // Width/height big-endian at offsets 16..24.
  const w = (bytes[16] << 24 | bytes[17] << 16 | bytes[18] << 8 | bytes[19]) >>> 0;
  const h = (bytes[20] << 24 | bytes[21] << 16 | bytes[22] << 8 | bytes[23]) >>> 0;
  assert.equal(w, 4); assert.equal(h, 3);
});

test('encodePng wraps a grayscale/alpha image consistently', () => {
  const img = rgba(5, 5, () => [0, 0, 0, 255]);
  const bytes = encodePng(img);
  assert.equal(bytes[0], 0x89);
  // The encoded stream must inflate back to w*h*4 bytes (RGBA by default).
  let idat = [];
  let off = 8;
  while (off < bytes.length) {
    const len = (bytes[off] << 24 | bytes[off + 1] << 16 | bytes[off + 2] << 8 | bytes[off + 3]) >>> 0;
    const type = String.fromCharCode(bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]);
    if (type === 'IDAT') idat.push(bytes.subarray(off + 8, off + 8 + len));
    off += 12 + len;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  assert.ok(raw.length >= 5 * (5 * 4 + 1), 'inflated scanlines carry the RGBA rows');
});
