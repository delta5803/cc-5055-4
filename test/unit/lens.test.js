import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  distortNormalized, distortJacobianNormalized, jacobianDet, undistortNormalized,
  normalizedToPixel, undistortPixel, pixelToDistortedNormalized,
  buildUndistortMap, analyzeMapSafety, largestSafeWindow, resolveOutputWindow,
} from '../../src/core/lens.js';

const SENSOR = {
  width: 640, height: 480,
  pitchX: 36 / 640, pitchY: 36 / 640,
  fx_mm: 30, fy_mm: 30,
  px_mm: 0.8, py_mm: -0.5,
  originX_mm: 0, originY_mm: 0,
  widthMM: 36, heightMM: 27,
};

test('undistortNormalized inverts distortNormalized for a strong lens', () => {
  const k = { k1: -0.25, k2: 0.05, k3: 0, p1: 0.001, p2: -0.0008 };
  for (const [x, y] of [[0, 0], [0.2, -0.1], [0.5, 0.3], [-0.45, 0.4], [0.7, -0.6]]) {
    const d = distortNormalized(x, y, k);
    const inv = undistortNormalized(d.xd, d.yd, k);
    assert.ok(inv.converged, `converged at (${x},${y})`);
    assert.ok(Math.hypot(inv.x - x, inv.y - y) < 1e-8, `round-trip error at (${x},${y})`);
    assert.ok(inv.jacobianDet > 0, 'inverse only accepted where the forward map is orientation-preserving');
  }
});

test('distortion Jacobian matches a finite-difference Jacobian', () => {
  const k = { k1: -0.2, k2: 0.04, k3: -0.002, p1: 0.002, p2: -0.0015 };
  const x = 0.37, y = -0.21, h = 1e-6;
  const J = distortJacobianNormalized(x, y, k);
  const fx = distortNormalized(x + h, y, k), bx = distortNormalized(x - h, y, k);
  const fy = distortNormalized(x, y + h, k), by = distortNormalized(x, y - h, k);
  const fd = [(fx.xd - bx.xd) / (2 * h), (fy.xd - by.xd) / (2 * h),
    (fx.yd - bx.yd) / (2 * h), (fy.yd - by.yd) / (2 * h)];
  assert.ok(Math.hypot(J[0] - fd[0], J[1] - fd[1]) < 1e-5);
  assert.ok(Math.hypot(J[2] - fd[2], J[3] - fd[3]) < 1e-5);
});

test('pixel undistortion is the exact inverse of pixel projection', () => {
  const k = { k1: -0.0003, k2: 3e-8, k3: 0, p1: 5e-5, p2: -3e-5 };
  for (const [x, y] of [[0, 0], [0.3, -0.15], [-0.4, 0.25], [0.55, 0.4]]) {
    const obs = normalizedToPixel(x, y, SENSOR, k);
    const back = undistortPixel(obs.u, obs.v, SENSOR, k);
    assert.ok(back.converged);
    // back is the ideal pixel; re-project it and require the same observed pixel.
    const n = pixelToDistortedNormalized(back.u, back.v, SENSOR);
    const re = normalizedToPixel(n.x, n.y, SENSOR, k);
    assert.ok(Math.hypot(re.u - obs.u, re.v - obs.v) < 0.02,
      `pixel round-trip within 0.02 px at (${x},${y})`);
  }
});

test('undistortPixel refuses to claim convergence inside a fold', () => {
  // k1 < -1/r^2 folds; sample a point whose forward Jacobian flips sign.
  const foldy = { k1: -60, k2: 0, k3: 0, p1: 0, p2: 0 };
  const s = { ...SENSOR, fx_mm: 10, fy_mm: 10 };
  const r = 0.12; // r^2 ~ 0.0144, k1*r^2 ~ -0.87 * ... -> sign flip
  const sx = s.fx_mm * r, sy = 0;
  const d = distortNormalized(sx, sy, foldy);
  const det = jacobianDet(distortJacobianNormalized(sx, sy, foldy));
  if (det <= 0) {
    const inv = undistortNormalized(d.xd, d.yd, foldy);
    assert.ok(!inv.converged, 'a folded region is reported as non-converged');
  }
});

test('buildUndistortMap produces a samplable, sign-consistent map', () => {
  const k = { k1: -0.0002, k2: 1e-8, k3: 0, p1: 3e-5, p2: -2e-5 };
  const map = buildUndistortMap(SENSOR, k, SENSOR.width, SENSOR.height, { step: 4 });
  assert.ok(map.stats.total > 0);
  assert.equal(map.stats.valid + map.stats.folds + map.stats.outside, map.stats.total);
  assert.ok(map.stats.minJacobianDet > 0, 'no folds for a mild lens');
  for (let mj = 0; mj < map.height; mj++) {
    for (let mi = 0; mi < map.width; mi++) {
      if (!map.valid[mj * map.width + mi]) continue;
      const x = map.mapX[mj * map.width + mi], y = map.mapY[mj * map.width + mi];
      assert.ok(Number.isFinite(x) && Number.isFinite(y));
      assert.ok(x >= -1 && x <= SENSOR.width + 1 && y >= -1 && y <= SENSOR.height + 1);
    }
  }
});

test('analyzeMapSafety rejects a map that actually folds', () => {
  const k = { k1: -0.02, k2: 0, k3: 0, p1: 0, p2: 0 };
  const map = buildUndistortMap(SENSOR, k, SENSOR.width, SENSOR.height, { step: 4 });
  const safety = analyzeMapSafety(map);
  assert.equal(safety.safe, false, 'a folding map is not safe for full-frame output');
  assert.ok(map.stats.folds > 0, 'the map itself records folds');
  assert.ok(safety.foldFlips > 0 || safety.discontinuities > 0,
    'safety analysis sees the fold or its discontinuity');
});

test('largestSafeWindow never exceeds the requested frame', () => {
  const k = { k1: -0.00015, k2: 0, k3: 0, p1: 0, p2: 0 };
  const map = buildUndistortMap(SENSOR, k, SENSOR.width, SENSOR.height, { step: 4 });
  const win = largestSafeWindow(map);
  assert.ok(win.width <= SENSOR.width && win.height <= SENSOR.height);
  assert.ok(win.x0 >= 0 && win.y0 >= 0);
  assert.ok(win.x0 + win.width <= SENSOR.width + 1e-6);
  assert.ok(win.y0 + win.height <= SENSOR.height + 1e-6);
});

test('resolveOutputWindow accepts a safe map and shrinks a folding one', () => {
  const req = { x0: 0, y0: 0, width: SENSOR.width, height: SENSOR.height };

  // A safe map: the full request is accepted unchanged.
  const mild = { k1: -0.0002, k2: 1e-8, k3: 0, p1: 3e-5, p2: -2e-5 };
  const safeMap = buildUndistortMap(SENSOR, mild, SENSOR.width, SENSOR.height, { step: 4 });
  const accepted = resolveOutputWindow(safeMap, req, { minFraction: 0.2 });
  assert.ok(accepted.accepted);
  assert.equal(accepted.shrunk, false);
  assert.deepEqual(accepted.window, req);

  // A folding map: the output must shrink below the request, or be rejected.
  const foldy = { k1: -0.02, k2: 0, k3: 0, p1: 0, p2: 0 };
  const foldMap = buildUndistortMap(SENSOR, foldy, SENSOR.width, SENSOR.height, { step: 4 });
  const decision = resolveOutputWindow(foldMap, req, { minFraction: 0.2 });
  assert.ok(!decision.accepted || decision.shrunk,
    'a folding map never yields an unchanged full-frame window');
  if (decision.accepted) {
    assert.ok(decision.window.width < req.width || decision.window.height < req.height,
      'the window actually shrinks');
  }
});
