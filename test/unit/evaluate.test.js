import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  defaultIdealLines, truthObservedPixel, heldOutLineBending, resamplingError,
  inverseMapHealth, centerOffset, gridLineBending, heldOutCornerError,
} from '../../src/core/evaluate.js';
import { normalizedToPixel, sensorXFromPixel, sensorYFromPixel } from '../../src/core/lens.js';

// A truth camera expressed in the sensor-mm convention the solvers use.
const TRUTH = {
  width: 640, height: 480,
  pitchX: 36 / 640, pitchY: 36 / 640,
  fx_mm: 30, fy_mm: 30,
  px_mm: 0.8, py_mm: -0.5,
  originX_mm: 0, originY_mm: 0,
  widthMM: 36,
  distortion: { k1: -0.0003, k2: 1.2e-7, k3: 0, p1: 3.4e-5, p2: -1.2e-5 },
};

// The same model, exactly, so every metric must report ~zero error.
const MODEL = { ...TRUTH, distortion: undefined };

test('centerOffset is zero against the same principal point and nonzero when shifted', () => {
  const zero = centerOffset(TRUTH, { px: TRUTH.px_mm, py: TRUTH.py_mm }, TRUTH.pitchX, TRUTH.pitchY);
  assert.ok(zero.distanceMM < 1e-12 && zero.distancePx < 1e-12);
  const shifted = centerOffset({ ...TRUTH, px_mm: TRUTH.px_mm + 0.1, py_mm: TRUTH.py_mm },
    { px: TRUTH.px_mm, py: TRUTH.py_mm }, TRUTH.pitchX, TRUTH.pitchY);
  assert.ok(Math.abs(shifted.dxmm - 0.1) < 1e-12);
  assert.ok(Math.abs(shifted.dxPx - 0.1 / TRUTH.pitchX) < 1e-9);
});

test('truthObservedPixel is the exact forward model', () => {
  const p = truthObservedPixel(320, 240, TRUTH);
  const xn = (sensorXFromPixel(320, TRUTH) - TRUTH.px_mm) / TRUTH.fx_mm;
  const yn = (sensorYFromPixel(240, TRUTH) - TRUTH.py_mm) / TRUTH.fy_mm;
  const exact = normalizedToPixel(xn, yn, TRUTH, TRUTH.distortion);
  assert.ok(Math.hypot(p.u - exact.u, p.v - exact.v) < 1e-9);
});

test('heldOutLineBending is ~0 when the candidate equals the truth lens', () => {
  const bend = heldOutLineBending(MODEL, TRUTH.distortion, TRUTH, TRUTH.width, TRUTH.height, { count: 6 });
  assert.ok(bend.lines > 0, 'some held-out lines were usable');
  assert.ok(bend.maxDeviationPx < 0.05, `exact model leaves ${bend.maxDeviationPx.toFixed(3)} px bend`);
});

test('heldOutLineBending reports real bending for a wrong model', () => {
  // Drop the radial term: the true image is bent, the candidate is not.
  const wrong = { k1: 0, k2: 0, k3: 0, p1: 0, p2: 0 };
  const bend = heldOutLineBending(MODEL, wrong, TRUTH, TRUTH.width, TRUTH.height, { count: 6 });
  assert.ok(bend.maxDeviationPx > 1, `missing distortion bends lines by ${bend.maxDeviationPx.toFixed(2)} px`);
});

test('resamplingError is ~0 when model and truth agree', () => {
  const win = { x0: 20, y0: 10, width: 600, height: 460 };
  const re = resamplingError(MODEL, TRUTH.distortion, TRUTH, win, { step: 4 });
  assert.ok(re.compared > 0);
  assert.ok(re.rmsPx < 1e-6, `identity model resamples to ${re.rmsPx.toExponential(2)} px`);
  assert.equal(re.folds, 0);
});

test('resamplingError grows when the model distortion is wrong', () => {
  const win = { x0: 20, y0: 10, width: 600, height: 460 };
  const wrong = { k1: 0, k2: 0, k3: 0, p1: 0, p2: 0 };
  const re = resamplingError(MODEL, wrong, TRUTH, win, { step: 4 });
  assert.ok(re.rmsPx > 0.5, `ignoring distortion costs ${re.rmsPx.toFixed(2)} px`);
});

test('inverseMapHealth: exact model folds nowhere over the frame', () => {
  const health = inverseMapHealth(MODEL, TRUTH.distortion, TRUTH.width, TRUTH.height, 25);
  assert.equal(health.folds, 0);
  assert.equal(health.unstable, 0);
  assert.ok(health.maxRoundTripPx < 0.05, 'round-trip is sub-pixel');
});

test('inverseMapHealth detects folds for an over-strong radial term', () => {
  const foldy = { k1: -0.02, k2: 0, k3: 0, p1: 0, p2: 0 };
  const health = inverseMapHealth(MODEL, foldy, TRUTH.width, TRUTH.height, 25);
  assert.ok(health.failureFraction > 0, 'a folding model is not reported healthy');
  // The fold (or the source it maps outside the frame) must appear somewhere.
  assert.ok(health.folds + health.outside > 0, 'the fold shows up as a fold or uncovered source');
});

test('gridLineBending is ~0 for a perfect lattice through the exact model', () => {
  // Build an ideal lattice and forward-distort it with the TRUTH lens; after
  // undistorting with the same lens the rows must be projective images of
  // straight lattice lines, i.e. perfectly straight.
  const views = [];
  for (let view = 0; view < 2; view++) {
    const gridIJ = [], imagePoints = [];
    const rot = view === 0 ? 0.02 : -0.03;
    for (let j = 0; j < 6; j++) {
      for (let i = 0; i < 9; i++) {
        // simple projective spread of the lattice across the frame
        const a = (i - 4) * 45 + 320 + view * 5;
        const b = (j - 2.5) * 42 + 240;
        const u = a + rot * b;
        const v = b + rot * a;
        const xn = (u - TRUTH.width / 2) * TRUTH.pitchX / TRUTH.fx_mm;
        const yn = (v - TRUTH.height / 2) * TRUTH.pitchY / TRUTH.fy_mm;
              const p = normalizedToPixel(xn, yn, TRUTH, TRUTH.distortion);
        gridIJ.push([i, j]); imagePoints.push([p.u, p.v]);
      }
    }
    views.push({ gridIJ, imagePoints });
  }
  const bend = gridLineBending(MODEL, TRUTH.distortion, views);
  assert.ok(bend.lines > 0);
  assert.ok(bend.maxDeviationPx < 0.05, `exact lens keeps rows straight (${bend.maxDeviationPx.toFixed(3)} px)`);
});

test('heldOutCornerError is ~0 for corners generated by the exact lens', () => {
  const gridIJ = [], imagePoints = [];
  for (let j = 0; j < 6; j++) for (let i = 0; i < 9; i++) {
    const a = (i - 4) * 40 + 320, b = (j - 2.5) * 40 + 240;
    const xn = (a - TRUTH.width / 2) * TRUTH.pitchX / TRUTH.fx_mm;
    const yn = (b - TRUTH.height / 2) * TRUTH.pitchY / TRUTH.fy_mm;
      const p = normalizedToPixel(xn, yn, TRUTH, TRUTH.distortion);
    gridIJ.push([i, j]); imagePoints.push([p.u, p.v]);
  }
  const err = heldOutCornerError(MODEL, TRUTH.distortion, [{ gridIJ, imagePoints }]);
  assert.ok(err.views.length === 1);
  assert.ok(err.meanPx < 0.02, `held-out corners reproject to ${err.meanPx.toFixed(3)} px`);
});

test('defaultIdealLines stay inside the image', () => {
  const lines = defaultIdealLines(640, 480, 5);
  assert.ok(lines.length >= 10);
  for (const l of lines) {
    for (const p of [l.p0, l.p1]) {
      assert.ok(p[0] >= 0 && p[0] <= 639 && p[1] >= 0 && p[1] <= 479);
    }
  }
});
