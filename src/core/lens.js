// Physical-sensor lens model.
//
// All geometry is expressed on the sensor surface in millimetres. A sensor
// description `s` has:
//
//   width, height          image size in pixels
//   pitchX, pitchY         mm per pixel (fixed for a given sensor)
//   fx_mm, fy_mm           focal length in mm
//   px_mm, py_mm           principal point in ABSOLUTE sensor coordinates
//   originX_mm, originY_mm centre of THIS image window in absolute sensor
//                          coordinates (0,0 for a full-frame readout, non-zero
//                          when the image is a crop of the sensor)
//
// Absolute sensor coordinates are defined with the origin at the sensor
// centre; pixel u maps to
//
//   Sx(u) = originX_mm + (u - (width - 1)/2) * pitchX
//
// and back with u = (Sx - originX_mm)/pitchX + (width - 1)/2.
//
// This is what lets a zoom + crop sequence share one lens: when the camera
// crops, `originX_mm` changes, so the same physical principal point appears
// at a different pixel. Solving per focal length ignores all of this and fits
// each image's intrinsics independently.
//
// Distortion D = { k1,k2,k3,p1,p2 } lives in sensor mm: it maps an ideal
// sensor offset from the principal point (fx_mm * X/Z, fy_mm * Y/Z) to the
// observed offset. Its coefficients therefore have units 1/mm^(2i) for the
// radial terms and 1/mm for the tangential ones, and are focal-length
// independent. The inverse is a Newton solve so folds can be detected.

// Distortion is evaluated in sensor-mm, so r^2 reaches hundreds. The clamp
// only guards the Newton iteration far outside the sensor; it must stay far
// above any physical radius or the Jacobian and the polynomial disagree.
const MAX_R2 = 1e8;

// ---------------------------------------------------------------------------
// Sensor description helpers
// ---------------------------------------------------------------------------

// Build a sensor description from pixel-space intrinsics. `sensor` may carry
// { widthMM, heightMM, pitchX, pitchY, originX_mm, originY_mm }. When pitch is
// supplied it wins: a cropped readout has fewer pixels but the SAME pitch, so
// deriving pitch from widthMM/cropWidth would be wrong.
export function intrinsicsToSensor(K, sensor = {}) {
  const { fx, fy, cx, cy, width, height } = K;
  const sensorW = sensor.widthMM ?? 36;
  const sensorH = sensor.heightMM ?? (sensorW * height) / width;
  const pitchX = sensor.pitchX ?? sensorW / width;
  const pitchY = sensor.pitchY ?? sensorH / height;
  const originX_mm = sensor.originX_mm ?? 0;
  const originY_mm = sensor.originY_mm ?? 0;
  return {
    width, height, pitchX, pitchY,
    fx_mm: fx * pitchX,
    fy_mm: fy * pitchY,
    px_mm: originX_mm + (cx - (width - 1) / 2) * pitchX,
    py_mm: originY_mm + (cy - (height - 1) / 2) * pitchY,
    originX_mm,
    originY_mm,
    widthMM: sensorW,
    heightMM: sensorH,
  };
}

export function sensorToPixelIntrinsics(s) {
  return {
    fx: s.fx_mm / s.pitchX,
    fy: s.fy_mm / s.pitchY,
    cx: (s.px_mm - s.originX_mm) / s.pitchX + (s.width - 1) / 2,
    cy: (s.py_mm - s.originY_mm) / s.pitchY + (s.height - 1) / 2,
    width: s.width,
    height: s.height,
  };
}

export function sensorXFromPixel(u, s) {
  return s.originX_mm + (u - (s.width - 1) / 2) * s.pitchX;
}
export function sensorYFromPixel(v, s) {
  return s.originY_mm + (v - (s.height - 1) / 2) * s.pitchY;
}
export function pixelXFromSensor(X, s) {
  return (X - s.originX_mm) / s.pitchX + (s.width - 1) / 2;
}
export function pixelYFromSensor(Y, s) {
  return (Y - s.originY_mm) / s.pitchY + (s.height - 1) / 2;
}

// ---------------------------------------------------------------------------
// Distortion polynomial (sensor mm)
// ---------------------------------------------------------------------------

export function defaultDistortion() {
  return { k1: 0, k2: 0, k3: 0, p1: 0, p2: 0 };
}

// Distort ideal normalised coordinates (x, y) = (X/Z, Y/Z) with coefficients
// that operate in the same units as `k`. For sensor-mm coefficients the
// caller passes fx_mm * x and fy_mm * y. Returns normalised-scale offsets.
export function distortNormalized(x, y, k) {
  let r2 = x * x + y * y;
  if (r2 > MAX_R2) r2 = MAX_R2;
  const r4 = r2 * r2;
  const r6 = r4 * r2;
  const radial = 1 + k.k1 * r2 + k.k2 * r4 + k.k3 * r6;
  const xd = x * radial + 2 * k.p1 * x * y + k.p2 * (r2 + 2 * x * x);
  const yd = y * radial + k.p1 * (r2 + 2 * y * y) + 2 * k.p2 * x * y;
  return { xd, yd };
}

export function distortJacobianNormalized(x, y, k) {
  const r2 = x * x + y * y;
  const r4 = r2 * r2;
  const r6 = r4 * r2;
  const radial = 1 + k.k1 * r2 + k.k2 * r4 + k.k3 * r6;
  const dradial = k.k1 + 2 * k.k2 * r2 + 3 * k.k3 * r4;
  const j00 = radial + 2 * x * x * dradial + 2 * k.p1 * y + 6 * k.p2 * x;
  const j01 = 2 * x * y * dradial + 2 * k.p1 * x + 2 * k.p2 * y;
  const j10 = 2 * x * y * dradial + 2 * k.p1 * x + 2 * k.p2 * y;
  const j11 = radial + 2 * y * y * dradial + 6 * k.p1 * y + 2 * k.p2 * x;
  return [j00, j01, j10, j11];
}

export function jacobianDet(J) {
  return J[0] * J[3] - J[1] * J[2];
}

export function undistortNormalized(xd, yd, k, opts = {}) {
  const maxIter = opts.maxIter ?? 40;
  const tol = opts.tol ?? 1e-14;
  let x = xd;
  let y = yd;
  for (let it = 0; it < maxIter; it++) {
    const f = distortNormalized(x, y, k);
    const ex = f.xd - xd;
    const ey = f.yd - yd;
    const J = distortJacobianNormalized(x, y, k);
    const det = jacobianDet(J);
    if (ex * ex + ey * ey < tol * tol) {
      return { x, y, converged: true, iterations: it, jacobianDet: det };
    }
    if (!(Math.abs(det) > 1e-14)) {
      return { x, y, converged: false, iterations: it, jacobianDet: det };
    }
    const dx = (J[3] * ex - J[1] * ey) / det;
    const dy = (-J[2] * ex + J[0] * ey) / det;
    x -= dx;
    y -= dy;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return { x, y, converged: false, iterations: it, jacobianDet: det };
    }
    if (dx * dx + dy * dy < tol * tol) {
      const Jf = distortJacobianNormalized(x, y, k);
      return { x, y, converged: true, iterations: it + 1, jacobianDet: jacobianDet(Jf) };
    }
  }
  const Jf = distortJacobianNormalized(x, y, k);
  return { x, y, converged: false, iterations: maxIter, jacobianDet: jacobianDet(Jf) };
}

// ---------------------------------------------------------------------------
// Pixel <-> normalised
// ---------------------------------------------------------------------------

// Pixel -> distorted normalised (X/Z, Y/Z). Distorted sensor offset divided by
// the focal length.
export function pixelToDistortedNormalized(u, v, s) {
  return {
    x: (sensorXFromPixel(u, s) - s.px_mm) / s.fx_mm,
    y: (sensorYFromPixel(v, s) - s.py_mm) / s.fy_mm,
  };
}

export function pixelToIdealNormalized(u, v, s, k) {
  // Undistort in sensor-mm (the coefficient units), then convert the ideal
  // sensor offset back to normalised coordinates so the result is the exact
  // inverse of normalizedToPixel().
  const sx = sensorXFromPixel(u, s) - s.px_mm;
  const sy = sensorYFromPixel(v, s) - s.py_mm;
  const inv = undistortNormalized(sx, sy, k);
  return {
    x: inv.x / s.fx_mm,
    y: inv.y / s.fy_mm,
    converged: inv.converged,
    iterations: inv.iterations,
    jacobianDet: inv.jacobianDet,
  };
}

// Observed (raw) pixel -> ideal (undistorted) pixel in the same window.
export function undistortPixel(u, v, s, k) {
  const inv = pixelToIdealNormalized(u, v, s, k);
  return {
    u: pixelXFromSensor(s.px_mm + s.fx_mm * inv.x, s),
    v: pixelYFromSensor(s.py_mm + s.fy_mm * inv.y, s),
    converged: inv.converged,
    iterations: inv.iterations,
    jacobianDet: inv.jacobianDet,
  };
}

// Ideal normalised -> pixel. Distortion is applied to (fx_mm*x, fy_mm*y) so
// the coefficients are in sensor-mm units.
export function normalizedToPixel(x, y, s, k) {
  const sx = s.fx_mm * x;
  const sy = s.fy_mm * y;
  const d = distortNormalized(sx, sy, k);
  return {
    u: pixelXFromSensor(s.px_mm + d.xd, s),
    v: pixelYFromSensor(s.py_mm + d.yd, s),
    distortionDet: jacobianDet(distortJacobianNormalized(sx, sy, k)),
  };
}

export function projectPoint(X, Y, Z, s, k) {
  return normalizedToPixel(X / Z, Y / Z, s, k);
}

// ---------------------------------------------------------------------------
// Distortion field (visualisation)
// ---------------------------------------------------------------------------

export function distortionFieldAt(u, v, s, k) {
  const ideal = pixelToIdealNormalized(u, v, s, k);
  const fwd = normalizedToPixel(ideal.x, ideal.y, s, k);
  const cu = pixelXFromSensor(s.px_mm, s);
  const cv = pixelYFromSensor(s.py_mm, s);
  const r = Math.hypot(u - cu, v - cv);
  const du = fwd.u - u;
  const dv = fwd.v - v;
  return {
    converged: ideal.converged,
    fold: ideal.jacobianDet <= 0,
    jacobianDet: ideal.jacobianDet,
    roundTripErrPx: Math.hypot(du, dv),
    relative: r > 1e-9 ? Math.hypot(du, dv) / r : 0,
  };
}

// ---------------------------------------------------------------------------
// Resampling map
// ---------------------------------------------------------------------------
//
// A remap stage consumes, for each DESTINATION pixel of the undistorted
// output, the SOURCE pixel in the raw frame:
//
//   source = forwardLens(destination ideal pixel)
//
// That is the map built here. A destination sample is usable only if the
// forward map is locally orientation preserving (Jacobian det > 0) and the
// source coordinate lies inside the raw frame. The safety analysis runs on
// these arrays, never on a parametric promise.

export function buildUndistortMap(s, k, outWidth, outHeight, opts = {}) {
  const originX = opts.originX ?? 0;
  const originY = opts.originY ?? 0;
  const step = opts.step ?? 1;
  const W = Math.ceil(outWidth / step);
  const H = Math.ceil(outHeight / step);
  const mapX = new Float32Array(W * H);
  const mapY = new Float32Array(W * H);
  const valid = new Uint8Array(W * H);
  let validCount = 0, folds = 0, outside = 0, minDet = Infinity;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const ou = originX + i * step;
      const ov = originY + j * step;
      const xn = (sensorXFromPixel(ou, s) - s.px_mm) / s.fx_mm;
      const yn = (sensorYFromPixel(ov, s) - s.py_mm) / s.fy_mm;
      const fwd = normalizedToPixel(xn, yn, s, k);
      const idx = j * W + i;
      if (fwd.distortionDet < minDet) minDet = fwd.distortionDet;
      let ok = false;
      if (fwd.distortionDet <= 0) folds++;
      else if (fwd.u < -0.5 || fwd.v < -0.5 || fwd.u > s.width - 0.5 || fwd.v > s.height - 0.5) outside++;
      else { ok = true; validCount++; }
      mapX[idx] = fwd.u;
      mapY[idx] = fwd.v;
      valid[idx] = ok ? 1 : 0;
    }
  }
  return {
    width: W, height: H, step, originX, originY, mapX, mapY, valid,
    stats: { total: W * H, valid: validCount, folds, outside, minJacobianDet: minDet },
  };
}

export function analyzeMapSafety(map) {
  const { width: W, height: H, mapX, mapY, valid } = map;
  const step = map.step ?? 1;
  // Destination samples are `step` px apart, so their sources should differ by
  // roughly `step`. A jump much larger than that is an implicit fold or branch
  // jump even if the sign never changes.
  const jumpLimit = 4 * step;
  let invalid = 0;
  for (let i = 0; i < valid.length; i++) if (!valid[i]) invalid++;
  let rowFlips = 0, rowJumps = 0;
  for (let j = 0; j < H; j++) {
    let prev = 0;
    for (let i = 1; i < W; i++) {
      const a = j * W + i, b = a - 1;
      if (!valid[a] || !valid[b]) { prev = 0; continue; }
      const dx = mapX[a] - mapX[b];
      const sgn = Math.sign(dx);
      if (sgn !== 0) {
        if (prev !== 0 && sgn !== prev) rowFlips++;
        prev = sgn;
      }
      if (Math.abs(dx) > jumpLimit) rowJumps++;
    }
  }
  let colFlips = 0, colJumps = 0;
  for (let i = 0; i < W; i++) {
    let prev = 0;
    for (let j = 1; j < H; j++) {
      const a = j * W + i, b = a - W;
      if (!valid[a] || !valid[b]) { prev = 0; continue; }
      const dy = mapY[a] - mapY[b];
      const sgn = Math.sign(dy);
      if (sgn !== 0) {
        if (prev !== 0 && sgn !== prev) colFlips++;
        prev = sgn;
      }
      if (Math.abs(dy) > jumpLimit) colJumps++;
    }
  }
  return {
    invalidSamples: invalid,
    fractionInvalid: invalid / valid.length,
    foldFlips: rowFlips + colFlips,
    discontinuities: rowJumps + colJumps,
    safe: invalid === 0 && rowFlips === 0 && colFlips === 0 && rowJumps === 0 && colJumps === 0,
  };
}

// Largest fully-usable axis-aligned window, in destination pixel coordinates.
export function largestSafeWindow(map) {
  const { width: W, height: H, valid, step } = map;
  const jumpLimit = 4 * step;
  const bad = new Uint8Array(W * H);
  for (let i = 0; i < valid.length; i++) if (!valid[i]) bad[i] = 1;
  for (let j = 0; j < H; j++) {
    for (let i = 1; i < W; i++) {
      const a = j * W + i, b = a - 1;
      if (!valid[a] || !valid[b]) continue;
      const dx = map.mapX[a] - map.mapX[b];
      if (Math.sign(dx) <= 0 || Math.abs(dx) > jumpLimit) bad[a] = 1;
    }
  }
  for (let i = 0; i < W; i++) {
    for (let j = 1; j < H; j++) {
      const a = j * W + i, b = a - W;
      if (!valid[a] || !valid[b]) continue;
      const dy = map.mapY[a] - map.mapY[b];
      if (Math.sign(dy) <= 0 || Math.abs(dy) > jumpLimit) bad[a] = 1;
    }
  }
  const heights = new Int32Array(W);
  let best = { area: 0, x0: 0, y0: 0, x1: 0, y1: 0 };
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) heights[i] = bad[j * W + i] ? 0 : heights[i] + 1;
    const stack = [];
    for (let i = 0; i <= W; i++) {
      const h = i === W ? 0 : heights[i];
      while (stack.length && heights[stack[stack.length - 1]] >= h) {
        const top = stack.pop();
        const left = stack.length ? stack[stack.length - 1] + 1 : 0;
        const width = i - left;
        const height = heights[top];
        const area = width * height;
        if (area > best.area) {
          best = {
            area,
            x0: left * step,
            y0: (j - height + 1) * step,
            x1: Math.min((left + width) * step, map.width * step),
            y1: (j + 1) * step,
          };
        }
      }
      stack.push(i);
    }
  }
  return {
    x0: best.x0 + map.originX,
    y0: best.y0 + map.originY,
    x1: best.x1 + map.originX,
    y1: best.y1 + map.originY,
    width: Math.max(0, best.x1 - best.x0),
    height: Math.max(0, best.y1 - best.y0),
    areaPx: Math.max(0, best.x1 - best.x0) * Math.max(0, best.y1 - best.y0),
    gridArea: best.area,
  };
}

// Decide what to do with a requested output window given the actual map:
// accept it, shrink it to the largest safe sub-window, or reject the segment.
export function resolveOutputWindow(map, requested, opts = {}) {
  const minFraction = opts.minFraction ?? 0.35;
  const safety = analyzeMapSafety(map);
  if (safety.safe) {
    return { accepted: true, shrunk: false, window: { ...requested }, safety };
  }
  const win = largestSafeWindow(map);
  const fracW = win.width / requested.width;
  const fracH = win.height / requested.height;
  if (win.areaPx <= 0 || fracW < minFraction || fracH < minFraction) {
    return {
      accepted: false,
      reason: win.areaPx <= 0
        ? 'no fold-free, in-coverage window exists for this segment'
        : `safe window is only ${(100 * fracW).toFixed(0)}% x ${(100 * fracH).toFixed(0)}% of the request`,
      safety,
      best: win,
    };
  }
  return {
    accepted: true,
    shrunk: true,
    window: {
      x0: win.x0, y0: win.y0, width: win.width, height: win.height,
    },
    safety,
    best: win,
  };
}
