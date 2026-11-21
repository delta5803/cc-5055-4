// Resampling + map export.
//
// The map produced by lens.buildUndistortMap maps DESTINATION (ideal) pixel ->
// SOURCE (raw) pixel. This module applies that map to an actual RGBA buffer so
// what the user exports is the same data the safety analysis ran on. It is
// deliberately browser-safe (no Node imports); PNG encoding lives in the UI.

function sampleBilinear(data, w, h, x, y, out, oi, channels = 4) {
  if (x < 0 || y < 0 || x > w - 1 || y > h - 1) {
    for (let c = 0; c < channels; c++) out[oi + c] = 0;
    return false;
  }
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  const i00 = (y0 * w + x0) * channels, i10 = (y0 * w + x1) * channels;
  const i01 = (y1 * w + x0) * channels, i11 = (y1 * w + x1) * channels;
  for (let c = 0; c < channels; c++) {
    const top = data[i00 + c] * (1 - fx) + data[i10 + c] * fx;
    const bot = data[i01 + c] * (1 - fx) + data[i11 + c] * fx;
    out[oi + c] = Math.round(top * (1 - fy) + bot * fy);
  }
  return true;
}

// Apply a source map to an RGBA image. `map` is { width, height, step,
// originX, originY, mapX, mapY, valid }. The output covers `window`
// ({x0,y0,width,height}) in destination pixel coordinates. Destination pixels
// whose map sample is invalid become transparent black; this is deliberate so a
// bad pixel can never masquerade as a corrected one.
export function resampleImage(src, map, outWindow) {
  const { data, width: sw, height: sh } = src;
  const ow = Math.round(outWindow.width);
  const oh = Math.round(outWindow.height);
  const out = new Uint8ClampedArray(ow * oh * 4);
  const { mapX, mapY, valid, step, width: mw, height: mh } = map;
  // outWindow is in the same (absolute destination) pixel coordinates the map
  // was built in; index 0 of the map arrays sits at (originX, originY).
  const ox = map.originX ?? 0;
  const oy = map.originY ?? 0;
  let sampled = 0, blank = 0;
  for (let j = 0; j < oh; j++) {
    const dv = outWindow.y0 + j;
    const mj = Math.floor((dv - oy) / step);
    for (let i = 0; i < ow; i++) {
      const du = outWindow.x0 + i;
      const mi = Math.floor((du - ox) / step);
      const oi = (j * ow + i) * 4;
      if (mi < 0 || mj < 0 || mi >= mw || mj >= mh || !valid[mj * mw + mi]) {
        out[oi] = 0; out[oi + 1] = 0; out[oi + 2] = 0; out[oi + 3] = 0;
        blank++;
        continue;
      }
      const ok = sampleBilinear(data, sw, sh, mapX[mj * mw + mi], mapY[mj * mw + mi], out, oi);
      if (ok) sampled++; else blank++;
    }
  }
  return { data: out, width: ow, height: oh, x0: outWindow.x0, y0: outWindow.y0, sampled, blank };
}

export function encodePng(image) {
  // Minimal uncompressed-deflate PNG writer: works in browser and Node, and a
  // correction map export should not depend on an optional image library.
  return encodePngRGBA(image.data, image.width, image.height);
}

function crc32(buf) {
  let c, crc = 0xffffffff;
  for (let n = 0; n < buf.length; n++) {
    c = (crc ^ buf[n]) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function adler32(buf) {
  let a = 1, b = 0;
  for (let i = 0; i < buf.length; i++) { a = (a + buf[i]) % 65521; b = (b + a) % 65521; }
  return ((b << 16) | a) >>> 0;
}

function deflateStore(buf) {
  // zlib header (no compression), stored blocks, adler32 trailer.
  const blocks = Math.max(1, Math.ceil(buf.length / 65535));
  const out = new Uint8Array(2 + blocks * 5 + buf.length + 4);
  out[0] = 0x78; out[1] = 0x01;
  let p = 2;
  for (let i = 0; i < blocks; i++) {
    const start = i * 65535;
    const len = Math.min(65535, buf.length - start);
    out[p++] = i === blocks - 1 ? 1 : 0;
    out[p++] = len & 0xff; out[p++] = (len >> 8) & 0xff;
    out[p++] = ~len & 0xff; out[p++] = (~len >> 8) & 0xff;
    out.set(buf.subarray(start, start + len), p);
    p += len;
  }
  const ad = adler32(buf);
  out[p++] = (ad >>> 24) & 0xff; out[p++] = (ad >>> 16) & 0xff;
  out[p++] = (ad >>> 8) & 0xff; out[p++] = ad & 0xff;
  return out.subarray(0, p);
}

export function encodePngRGBA(rgba, width, height) {
  // Raw scanlines with filter byte 0.
  const stride = width * 4;
  const raw = new Uint8Array((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0;
    raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
  }
  const z = deflateStore(raw);
  const chunks = [];
  const sig = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  chunks.push(sig);
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width); dv.setUint32(4, height);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  chunks.push(makeChunk('IHDR', ihdr));
  chunks.push(makeChunk('IDAT', z));
  chunks.push(makeChunk('IEND', new Uint8Array(0)));
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function makeChunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  const crcInput = out.subarray(4, 8 + data.length);
  dv.setUint32(8 + data.length, crc32(crcInput));
  return out;
}

// Sub-sampled map dump so the exported JSON stays a reasonable size while
// still being an actually-consumable resampling map (step is recorded).
export function mapToJSON(map, extra = {}) {
  return {
    format: 'lens-undistort-map/1',
    width: map.width,
    height: map.height,
    step: map.step,
    originX: map.originX,
    originY: map.originY,
    stats: map.stats,
    ...extra,
    mapX: Array.from(map.mapX, (v) => Math.round(v * 1000) / 1000),
    mapY: Array.from(map.mapY, (v) => Math.round(v * 1000) / 1000),
    valid: Array.from(map.valid),
  };
}

// Source pixel for a destination pixel, with the same semantics the resampler
// uses (nearest map sample). Used by the UI's hover readout.
export function sourceAt(map, du, dv) {
  const mi = Math.floor((du - (map.originX ?? 0)) / map.step);
  const mj = Math.floor((dv - (map.originY ?? 0)) / map.step);
  if (mi < 0 || mj < 0 || mi >= map.width || mj >= map.height) return null;
  const idx = mj * map.width + mi;
  return {
    u: map.mapX[idx], v: map.mapY[idx], valid: !!map.valid[idx], index: idx,
  };
}
