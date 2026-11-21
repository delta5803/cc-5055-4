// Solve pipeline for the workbench.
//
// One call turns the current dataset state (which views are deleted, which
// corners are excluded, which views are held out) into everything the UI
// renders:
//
//   fit        per-segment lens parameters (either path)
//   map        the ACTUAL destination->source resampling map for each segment
//   window     what resolveOutputWindow decided from that map (accept/shrink/
//              reject) — never an average-residual guess
//   resampled  the undistorted raster for the selected view, produced by
//              applying the same map
//   metrics    truth checks (synthetic) or held-out checks (real)

import { solvePerFocalLength, solveSharedSensor, distortionMMToPixel } from '../core/calibrate.js';
import {
  buildUndistortMap, resolveOutputWindow, analyzeMapSafety, sensorToPixelIntrinsics,
} from '../core/lens.js';
import { resampleImage } from '../core/resample.js';
import {
  heldOutLineBending, resamplingError, inverseMapHealth, centerOffset,
  gridLineBending, heldOutCornerError,
} from '../core/evaluate.js';

// Detected corners carry integer lattice indices; the object-space scale only
// matters for the synthetic truth (real photos use an arbitrary consistent
// scale, and every reported metric is in pixels).
export function viewCorrespondence(view, squareSize = 1) {
  if (!view.grid) return null;
  const obj = [];
  const img = [];
  for (let k = 0; k < view.grid.points.length; k++) {
    if (view.excluded?.has(k)) continue;
    const [i, j] = view.grid.gridIJ[k];
    obj.push([i * squareSize, j * squareSize, 0]);
    img.push(view.grid.points[k]);
  }
  if (obj.length < 4) return null;
  return { objectPoints: obj, imagePoints: img };
}

export function segmentViews(seg, squareSize, only = null) {
  const out = [];
  for (const v of seg.views) {
    if (v.deleted) continue;
    if (only && !only.includes(v)) continue;
    const c = viewCorrespondence(v, squareSize);
    if (c) out.push({ ...c, width: v.width, height: v.height, viewId: v.id });
  }
  return out;
}

// Solve one path across all segments and produce per-segment map/window data.
export function solve(dataset, opts = {}) {
  const squareSize = dataset.kind === 'synthetic' ? dataset.truth.board.squareSize : 1;
  const path = opts.path ?? 'shared';
  const model = opts.model ?? 'poly5';
  const requestFull = opts.windowMode === 'full';
  const minFraction = requestFull ? 1 : (opts.minFraction ?? 0.3);
  const sensorCfg = {
    widthMM: dataset.truth.sensorMM,
    heightMM: dataset.truth.sensorMM * (dataset.segments[0].height / dataset.segments[0].width),
    pitchX: dataset.truth.pitchX,
    pitchY: dataset.truth.pitchY,
  };

  // Each segment's fitting views (already filtered for deletion/exclusion).
  const fitViews = dataset.segments.map((seg) =>
    segmentViews(seg, squareSize, opts.fitViewsBySeg?.get(seg.focalId)));

  let perSegment = new Array(dataset.segments.length).fill(null);
  let shared = null;
  const errors = [];

  if (path === 'shared') {
    const payload = [];
    const payloadIndex = [];
    for (let i = 0; i < dataset.segments.length; i++) {
      if (fitViews[i].length < 2) {
        errors.push(`segment ${dataset.segments[i].focalId}: only ${fitViews[i].length} usable views (need >=2)`);
        continue;
      }
      payloadIndex.push(i);
      payload.push({
        focalId: dataset.segments[i].focalId,
        originX_mm: dataset.segments[i].originX_mm ?? 0,
        originY_mm: dataset.segments[i].originY_mm ?? 0,
        views: fitViews[i],
      });
    }
    if (payload.length === 0) return { ok: false, errors };
    try {
      shared = solveSharedSensor(payload, {
        sensor: sensorCfg,
        imageSize: { width: dataset.segments[0].width, height: dataset.segments[0].height },
        model,
      });
      shared.segments.forEach((s, k) => {
        const i = payloadIndex[k];
        perSegment[i] = {
          focalId: s.focalId,
          sensor: s.sensor,
          distortionMM: shared.distortionMM,
          rms: s.views.reduce((a, v) => a + v.rms, 0) / Math.max(1, s.views.length),
          perView: s.views,
          converged: shared.converged,
          model: shared.model,
          note: shared.note,
        };
      });
    } catch (e) {
      errors.push(`shared solve failed: ${e.message}`);
      return { ok: false, errors };
    }
  } else {
    for (let i = 0; i < dataset.segments.length; i++) {
      const seg = dataset.segments[i];
      const views = fitViews[i];
      if (views.length < 2) {
        errors.push(`segment ${seg.focalId}: only ${views.length} usable views (need >=2)`);
        continue;
      }
      try {
        const r = solvePerFocalLength(views, {
          sensor: {
            ...sensorCfg,
            originX_mm: seg.originX_mm ?? 0,
            originY_mm: seg.originY_mm ?? 0,
          },
          model,
        });
        perSegment[i] = {
          focalId: seg.focalId,
          sensor: r.sensor,
          distortionMM: r.distortionMM,
          rms: r.rms,
          perView: r.perView,
          converged: r.converged,
          model: r.model,
          note: r.note,
        };
      } catch (e) {
        errors.push(`segment ${seg.focalId}: ${e.message}`);
      }
    }
  }

  // Per-segment map, safety, output window and verification.
  const results = [];
  for (let si = 0; si < dataset.segments.length; si++) {
    const seg = dataset.segments[si];
    const fit = perSegment[si];
    if (!fit) { results.push(null); continue; }
    const W = seg.width, H = seg.height;
    const step = opts.mapStep ?? 2;
    const map = buildUndistortMap(fit.sensor, fit.distortionMM, W, H, { step });
    const request = { x0: 0, y0: 0, width: W, height: H };
    let decision;
    if (requestFull) {
      // The user explicitly asked for the whole frame. If the actual map folds
      // or leaves coverage anywhere, do not quietly trim it: reject the segment.
      const safety = analyzeMapSafety(map);
      decision = safety.safe
        ? { accepted: true, shrunk: false, window: { ...request }, safety }
        : {
          accepted: false,
          reason: 'full frame requested but the actual map folds or leaves source coverage',
          safety,
          best: null,
        };
    } else {
      decision = resolveOutputWindow(map, request, { minFraction });
    }
    const safety = analyzeMapSafety(map);
    const K = sensorToPixelIntrinsics(fit.sensor);
    const Dpx = distortionMMToPixel(fit.distortionMM, fit.sensor.fx_mm);
    const invHealth = inverseMapHealth(fit.sensor, fit.distortionMM, W, H, 25);

    let verification = null;
    if (dataset.kind === 'synthetic') {
      const truth = truthCameraFor(seg, dataset.truth);
      const co = centerOffset(fit.sensor, dataset.truth.principalMM,
        dataset.truth.pitchX, dataset.truth.pitchY);
      const lb = heldOutLineBending(fit.sensor, fit.distortionMM, truth, W, H);
      const win = decision.accepted
        ? decision.window
        : (decision.best ?? request);
      const re = resamplingError(fit.sensor, fit.distortionMM, truth, win, { step: 3 });
      verification = {
        kind: 'synthetic',
        center: co,
        lineBending: lb,
        resampling: re,
        truthDistortion: dataset.truth.distortion,
        truthPP: dataset.truth.principalMM,
      };
    } else {
      const allUsable = seg.views.filter((v) => v.grid && !v.deleted);
      const fitSet = opts.fitViewsBySeg?.get(seg.focalId);
      const held = fitSet ? allUsable.filter((v) => !fitSet.includes(v)) : [];
      const gl = gridLineBending(fit.sensor, fit.distortionMM,
        allUsable.map((v) => ({ gridIJ: v.grid.gridIJ, imagePoints: v.grid.points })));
      verification = {
        kind: 'real',
        gridBending: gl,
        heldOutViews: held.length,
        heldOutCorner: held.length
          ? heldOutCornerError(fit.sensor, fit.distortionMM,
            held.map((v) => ({ gridIJ: v.grid.gridIJ, imagePoints: v.grid.points })))
          : null,
      };
    }

    results.push({
      focalId: seg.focalId, segIndex: si,
      K, sensor: fit.sensor, distortionMM: fit.distortionMM, distortionPx: Dpx,
      rms: fit.rms, converged: fit.converged, model: fit.model, note: fit.note,
      perView: fit.perView,
      map, safety, decision, invHealth, verification,
      width: W, height: H,
    });
  }

  return { ok: true, path, model, shared, perSegment: results, errors };
}

// Reconstruct the exact truth camera used to render a segment.
export function truthCameraFor(seg, truthMeta) {
  return {
    width: seg.width, height: seg.height,
    pitchX: truthMeta.pitchX, pitchY: truthMeta.pitchY,
    fx_mm: seg.f_mm, fy_mm: seg.f_mm,
    px_mm: truthMeta.principalMM.px, py_mm: truthMeta.principalMM.py,
    originX_mm: seg.originX_mm ?? 0,
    originY_mm: seg.originY_mm ?? 0,
    distortion: truthMeta.distortion,
    widthMM: truthMeta.sensorMM,
  };
}

// Resample a view with a fitted map, over the accepted window. The returned
// image is in the destination (ideal) frame; `image.x0/y0` are its absolute
// origin so overlays can place it correctly.
export function resampleView(view, result) {
  const win = result.decision.accepted
    ? result.decision.window
    : (result.decision.best ?? { x0: 0, y0: 0, width: result.width, height: result.height });
  const x0 = Math.round(win.x0), y0 = Math.round(win.y0);
  const w = Math.max(1, Math.round(win.width));
  const h = Math.max(1, Math.round(win.height));
  // Build the map in the same absolute destination coordinates as the window,
  // at full resolution, so the exported JSON and the raster share one map.
  const map = buildUndistortMap(result.sensor, result.distortionMM, w, h,
    { originX: x0, originY: y0, step: 1 });
  const src = toRGBA(view.gray);
  const image = resampleImage(src, map, { x0, y0, width: w, height: h });
  return { image, map, window: { x0, y0, width: w, height: h } };
}

// Float32/float grayscale -> RGBA for resampling.
export function toRGBA(gray) {
  const { data, width, height } = gray;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i++) {
    const v = Math.round(Math.max(0, Math.min(1, data[i])) * 255);
    const j = i * 4;
    out[j] = v; out[j + 1] = v; out[j + 2] = v; out[j + 3] = 255;
  }
  return { data: out, width, height };
}
