// Verification metrics.
//
// These are the numbers the workbench must report honestly, all computed from
// the actual fitted model and the actual resampling map rather than from a
// parametric promise:
//
//  - centre offset: fitted principal point against the synthetic truth (mm/px)
//  - peripheral inverse-map health: converges? folds? out of source?
//  - held-out straight-line bending: lines that were never in the fit are
//    projected through the TRUE lens and straightened with the candidate model;
//    any remaining non-collinearity is error
//  - final resampling error: model source pixel vs the true lens source pixel
//    for the same destination ray, over the accepted output window
//
// For real photographs there is no truth camera, so the truth-based metrics
// are skipped and the corner-grid line straightness / held-out corner error are
// used instead.

import {
  normalizedToPixel, undistortPixel,
  sensorXFromPixel, sensorYFromPixel, buildUndistortMap,
} from './lens.js';
import { lineDeviation } from './synth.js';
import { homography } from './calibrate.js';

// Straight ideal-image lines. Every straight line in a pinhole image is the
// image of some physical straight line, so these are legitimate held-out
// geometric references without having to model rods in 3D.
export function defaultIdealLines(width, height, count = 9) {
  const lines = [];
  const W = width - 1, H = height - 1;
  const m = 0.04;
  for (let i = 0; i < count; i++) {
    const t = m + (1 - 2 * m) * (i / (count - 1));
    lines.push({ p0: [m * W, t * H], p1: [(1 - m) * W, t * H] });       // horizontal
    lines.push({ p0: [t * W, m * H], p1: [t * W, (1 - m) * H] });       // vertical
  }
  // A few diagonals to catch tangential/rotational error that axis-aligned
  // lines can miss.
  lines.push({ p0: [0.06 * W, 0.06 * H], p1: [0.94 * W, 0.94 * H] });
  lines.push({ p0: [0.06 * W, 0.94 * H], p1: [0.94 * W, 0.06 * H] });
  return lines;
}

// Observed pixel where the true lens images the ideal pixel (u, v).
export function truthObservedPixel(u, v, truth) {
  const xn = (sensorXFromPixel(u, truth) - truth.px_mm) / truth.fx_mm;
  const yn = (sensorYFromPixel(v, truth) - truth.py_mm) / truth.fy_mm;
  return normalizedToPixel(xn, yn, truth, truth.distortion);
}

// Project held-out ideal lines through the true lens and straighten them with
// the candidate model. The straightened points must be collinear.
export function heldOutLineBending(modelSensor, modelD, truth, width, height, opts = {}) {
  const lines = opts.lines ?? defaultIdealLines(width, height, opts.count ?? 9);
  const samples = opts.samples ?? 48;
  const perLine = [];
  let worst = 0, sum = 0, n = 0, used = 0;
  for (const line of lines) {
    const pts = [];
    for (let i = 0; i < samples; i++) {
      const s = i / (samples - 1);
      const u = line.p0[0] + s * (line.p1[0] - line.p0[0]);
      const v = line.p0[1] + s * (line.p1[1] - line.p0[1]);
      const obs = truthObservedPixel(u, v, truth);
      if (!(obs.u >= 0 && obs.v >= 0 && obs.u <= width - 1 && obs.v <= height - 1)) continue;
      const back = undistortPixel(obs.u, obs.v, modelSensor, modelD);
      if (!back.converged) continue;
      pts.push([back.u, back.v]);
    }
    if (pts.length < 8) continue;
    const dev = lineDeviation(pts);
    perLine.push({ maxDeviationPx: dev.maxDeviationPx, meanDeviationPx: dev.meanDeviationPx, samples: pts.length });
    worst = Math.max(worst, dev.maxDeviationPx);
    sum += dev.meanDeviationPx;
    n++;
    used += pts.length;
  }
  return {
    lines: perLine.length,
    maxDeviationPx: worst,
    meanDeviationPx: n ? sum / n : Infinity,
    samples: used,
  };
}

// Model source pixel vs true lens source pixel for every destination pixel in
// `window`. `window` is in the model's output pixel frame. Returns RMS/max in
// source pixels plus counts of unusable samples.
export function resamplingError(modelSensor, modelD, truth, window, opts = {}) {
  const step = opts.step ?? 2;
  const x0 = Math.max(0, Math.floor(window.x0));
  const y0 = Math.max(0, Math.floor(window.y0));
  const x1 = Math.ceil(window.x0 + window.width);
  const y1 = Math.ceil(window.y0 + window.height);
  let sumSq = 0, maxErr = 0, n = 0, folds = 0, outside = 0, compared = 0;
  for (let v = y0; v < y1; v += step) {
    for (let u = x0; u < x1; u += step) {
      n++;
      // Physical direction of this ideal destination pixel in model terms.
      const xn = (sensorXFromPixel(u, modelSensor) - modelSensor.px_mm) / modelSensor.fx_mm;
      const yn = (sensorYFromPixel(v, modelSensor) - modelSensor.py_mm) / modelSensor.fy_mm;
      const modelSrc = normalizedToPixel(xn, yn, modelSensor, modelD);
      if (!(modelSrc.distortionDet > 0)) { folds++; continue; }
      if (modelSrc.u < -0.5 || modelSrc.v < -0.5 || modelSrc.u > modelSensor.width - 0.5 || modelSrc.v > modelSensor.height - 0.5) {
        outside++;
        continue;
      }
      // Same physical direction through the true camera.
      const truthSrc = normalizedToPixel(xn, yn, truth, truth.distortion);
      const e = Math.hypot(modelSrc.u - truthSrc.u, modelSrc.v - truthSrc.v);
      sumSq += e * e;
      if (e > maxErr) maxErr = e;
      compared++;
    }
  }
  return {
    rmsPx: compared ? Math.sqrt(sumSq / compared) : Infinity,
    maxPx: maxErr,
    samples: n,
    compared,
    folds,
    outside,
  };
}

// Peripheral inverse-map health over the whole image. This follows exactly the
// same semantics as buildUndistortMap: a DESTINATION pixel is an ideal pixel;
// forward-map it to a source pixel, then confirm the inverse solve recovers the
// same ideal pixel. A destination is unhealthy if the forward map folds
// (non-orientation-preserving), leaves the raw frame, or cannot be inverted.
export function inverseMapHealth(modelSensor, modelD, width, height, grid = 33) {
  let folds = 0, outside = 0, unstable = 0, edgeFolds = 0, n = 0;
  let maxRound = 0;
  for (let j = 0; j < grid; j++) {
    for (let i = 0; i < grid; i++) {
      const u = (i / (grid - 1)) * (width - 1);
      const v = (j / (grid - 1)) * (height - 1);
      n++;
      const edge = Math.max(Math.abs(u - (width - 1) / 2) / ((width - 1) / 2),
        Math.abs(v - (height - 1) / 2) / ((height - 1) / 2)) > 0.75;
      const xn = (sensorXFromPixel(u, modelSensor) - modelSensor.px_mm) / modelSensor.fx_mm;
      const yn = (sensorYFromPixel(v, modelSensor) - modelSensor.py_mm) / modelSensor.fy_mm;
      const src = normalizedToPixel(xn, yn, modelSensor, modelD);
      if (!(src.distortionDet > 0)) {
        folds++;
        if (edge) edgeFolds++;
        continue;
      }
      if (src.u < -0.5 || src.v < -0.5 || src.u > modelSensor.width - 0.5 || src.v > modelSensor.height - 0.5) {
        outside++;
        continue;
      }
      // Invert the source pixel back to an ideal pixel; it must reproduce
      // (u, v). undistortPixel returns ideal PIXEL coordinates directly.
      const back = undistortPixel(src.u, src.v, modelSensor, modelD);
      const e = Math.hypot(back.u - u, back.v - v);
      if (e > maxRound) maxRound = e;
      if (!back.converged || e > 0.25) unstable++;
    }
  }
  return {
    samples: n,
    folds,
    outside,
    unstable,
    edgeFolds,
    // Fold/unstable pixels are model failures; `outside` is merely uncovered
    // source area, which the safe-window logic handles.
    failureFraction: (folds + unstable) / n,
    maxRoundTripPx: maxRound,
  };
}

// Centre offset of the fitted principal point w.r.t. the truth, in mm and px.
export function centerOffset(modelSensor, truthPP, pitchX, pitchY) {
  const dxmm = modelSensor.px_mm - truthPP.px;
  const dymm = modelSensor.py_mm - truthPP.py;
  return {
    dxmm, dymm,
    distanceMM: Math.hypot(dxmm, dymm),
    dxPx: dxmm / pitchX,
    dyPx: dymm / pitchY,
    distancePx: Math.hypot(dxmm / pitchX, dymm / pitchY),
  };
}

// Straightness of the detected corner rows/columns after undistortion. Used on
// real photographs (no truth camera): each row of the board's lattice lies on
// a physical straight line, so after a correct undistortion it must be straight.
export function gridLineBending(modelSensor, modelD, views) {
  const maxDev = [];
  for (const v of views) {
    if (!v.gridIJ) continue;
    const byRow = new Map(), byCol = new Map();
    for (let k = 0; k < v.gridIJ.length; k++) {
      const [i, j] = v.gridIJ[k];
      const u = v.imagePoints[k][0], w = v.imagePoints[k][1];
      const p = undistortPixel(u, w, modelSensor, modelD);
      if (!p.converged) continue;
      if (!byRow.has(j)) byRow.set(j, []);
      if (!byCol.has(i)) byCol.set(i, []);
      byRow.get(j).push([p.u, p.v]);
      byCol.get(i).push([p.u, p.v]);
    }
    for (const arr of [...byRow.values(), ...byCol.values()]) {
      if (arr.length < 4) continue;
      maxDev.push(lineDeviation(arr).maxDeviationPx);
    }
  }
  if (!maxDev.length) return { maxDeviationPx: Infinity, meanDeviationPx: Infinity, lines: 0 };
  return {
    maxDeviationPx: Math.max(...maxDev),
    meanDeviationPx: maxDev.reduce((a, b) => a + b, 0) / maxDev.length,
    lines: maxDev.length,
  };
}

// Held-out board check for real photographs.
//
// The board is planar and its lattice coordinates are known. If the lens model
// is right, undistorting every corner of a held-out view must map the lattice
// onto a single projective image (a homography). So: undistort the held-out
// corners, fit one homography lattice -> ideal pixel, and report the residuals.
// A model that only fits the views it saw will leave structure here.
export function heldOutCornerError(modelSensor, modelD, views) {
  const perView = [];
  let worst = 0, sum = 0, n = 0;
  for (const v of views) {
    if (!v.gridIJ) continue;
    const src = [];
    const dst = [];
    for (let k = 0; k < v.gridIJ.length; k++) {
      const [i, j] = v.gridIJ[k];
      const u = v.imagePoints[k][0], w = v.imagePoints[k][1];
      const p = undistortPixel(u, w, modelSensor, modelD);
      if (!p.converged) continue;
      src.push([i, j]);
      dst.push([p.u, p.v]);
    }
    if (src.length < 4) continue;
    const H = homography(src, dst);
    let sq = 0, mx = 0, cnt = 0;
    for (let k = 0; k < src.length; k++) {
      const pr = applyH(H, src[k][0], src[k][1]);
      const e = Math.hypot(pr[0] - dst[k][0], pr[1] - dst[k][1]);
      sq += e * e; mx = Math.max(mx, e); cnt++;
    }
    const rms = Math.sqrt(sq / cnt);
    perView.push({ rmsPx: rms, maxPx: mx, corners: cnt });
    sum += sq; n += cnt; worst = Math.max(worst, mx);
  }
  return {
    meanPx: n ? Math.sqrt(sum / n) : Infinity,
    maxPx: worst,
    corners: n,
    views: perView,
  };
}

function applyH(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ];
}

// Build a compact summary of a map for the UI.
export function mapSummary(map, request) {
  const fractionValid = map.stats.valid / map.stats.total;
  return {
    valid: map.stats.valid,
    total: map.stats.total,
    folds: map.stats.folds,
    outside: map.stats.outside,
    fractionValid,
    minJacobianDet: map.stats.minJacobianDet,
    requested: request,
  };
}

export { buildUndistortMap };
