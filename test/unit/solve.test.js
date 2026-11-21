import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadSynthetic } from '../../src/app/dataset.js';
import { solve, resampleView, viewCorrespondence, segmentViews } from '../../src/app/solve.js';

// The solve pipeline is where the task's promises are actually kept: two paths
// over the same observed views, a map built from the fitted model, a window
// derived from that map (not from an average residual), and resampling that
// uses the exported map. These tests exercise it headlessly.

let dataset;
function ds() {
  if (!dataset) dataset = loadSynthetic();
  return dataset;
}

test('viewCorrespondence drops excluded corners', () => {
  const view = ds().segments[0].views[0];
  assert.ok(view.grid, 'detector found the board');
  const all = viewCorrespondence(view, 1);
  assert.equal(all.objectPoints.length, view.grid.points.length);
  const before = view.excluded;
  view.excluded = new Set([0, 1, 2]);
  try {
    const trimmed = viewCorrespondence(view, 1);
    assert.equal(trimmed.objectPoints.length, all.objectPoints.length - 3);
  } finally {
    view.excluded = before;
  }
});

test('segmentViews omits deleted views', () => {
  const seg = ds().segments[0];
  const all = segmentViews(seg, 1);
  const victim = seg.views[0];
  victim.deleted = true;
  try {
    const after = segmentViews(seg, 1);
    assert.equal(after.length, all.length - 1);
    assert.ok(!after.some((v) => v.viewId === victim.id));
  } finally {
    victim.deleted = false;
  }
});

test('solve produces a map, an actual-map window and truth verification', () => {
  const r = solve(ds(), { path: 'shared', model: 'poly5', mapStep: 4 });
  assert.ok(r.ok, r.errors.join('; '));
  for (const seg of r.perSegment) {
    assert.ok(seg, 'segment solved');
    // The map is real: its stats must account for every sample.
    const st = seg.map.stats;
    assert.equal(st.valid + st.folds + st.outside, st.total);
    // The decision came from that map.
    assert.ok(seg.safety && typeof seg.safety.safe === 'boolean');
    assert.ok(seg.decision.accepted === true || seg.decision.accepted === false);
    if (seg.decision.accepted) {
      assert.ok(seg.decision.window.width > 0 && seg.decision.window.height > 0);
    } else {
      assert.ok(seg.decision.best === null || seg.decision.best.width >= 0);
    }
    // Synthetic truth checks are present and honest (no NaN).
    const v = seg.verification;
    assert.equal(v.kind, 'synthetic');
    for (const k of ['center', 'lineBending', 'resampling']) assert.ok(v[k]);
    assert.ok(Number.isFinite(v.center.distancePx));
    assert.ok(Number.isFinite(v.lineBending.maxDeviationPx));
    assert.ok(Number.isFinite(v.resampling.rmsPx));
  }
});

test('per-focal solve also succeeds and reports per-segment parameters', () => {
  const r = solve(ds(), { path: 'per', model: 'poly5', mapStep: 4 });
  assert.ok(r.ok, r.errors.join('; '));
  assert.equal(r.perSegment.filter(Boolean).length, 3);
  for (const seg of r.perSegment) {
    if (!seg) continue;
    assert.ok(seg.sensor.fx_mm > 0);
    assert.ok(seg.rms < 2, `${seg.focalId} reprojection ${seg.rms.toFixed(3)} px`);
  }
});

test('deleting the only peripheral view shrinks the per-focal safe crop', () => {
  const d = loadSynthetic(); // fresh copy so mutation is local
  const base = solve(d, { path: 'per', model: 'poly5', mapStep: 4 });
  const wideBase = base.perSegment.find((s) => s && s.focalId === 'wide');
  assert.ok(wideBase.decision.accepted);
  const baseW = wideBase.decision.window.width;
  const baseH = wideBase.decision.window.height;

  // wide#4 is the single tilted view that reaches the sensor corners.
  d.segments[0].views[4].deleted = true;
  const after = solve(d, { path: 'per', model: 'poly5', mapStep: 4 });
  const wideAfter = after.perSegment.find((s) => s && s.focalId === 'wide');
  assert.ok(wideAfter.decision.accepted);
  const aw = wideAfter.decision.window.width;
  const ah = wideAfter.decision.window.height;

  assert.ok(aw < baseW || ah < baseH,
    `per-focal crop changes when the peripheral view is removed (${baseW}x${baseH} -> ${aw}x${ah})`);
});

test('deleting the same view leaves the shared crop anchored', () => {
  const d = loadSynthetic();
  const base = solve(d, { path: 'shared', model: 'poly5', mapStep: 4 });
  const wideBase = base.perSegment.find((s) => s && s.focalId === 'wide');
  const areaBase = wideBase.decision.window.width * wideBase.decision.window.height;

  d.segments[0].views[4].deleted = true;
  const after = solve(d, { path: 'shared', model: 'poly5', mapStep: 4 });
  const wideAfter = after.perSegment.find((s) => s && s.focalId === 'wide');
  const areaAfter = wideAfter.decision.window.width * wideAfter.decision.window.height;

  // The other segments anchor the shared distortion, so the crop survives.
  assert.ok(areaAfter > 0.85 * areaBase,
    `shared crop stays anchored (${areaBase} -> ${areaAfter} px^2)`);
});

test('full-frame mode rejects a segment whose actual map folds', () => {
  const d = loadSynthetic();
  d.segments[0].views[4].deleted = true; // induce a fold on the per-focal path
  const r = solve(d, { path: 'per', model: 'poly5', mapStep: 4, windowMode: 'full' });
  const wide = r.perSegment.find((s) => s && s.focalId === 'wide');
  // Either the map is genuinely safe (accepted) or the full request is
  // rejected with a reason — but it must never silently trim and call it full.
  if (!wide.decision.accepted) {
    assert.match(wide.decision.reason, /fold|coverage/i);
  } else {
    assert.equal(wide.decision.window.width, d.segments[0].width);
    assert.equal(wide.decision.window.height, d.segments[0].height);
  }
});

test('resampleView applies the exported map and leaves at most a thin blank border', () => {
  const d = loadSynthetic();
  const r = solve(d, { path: 'shared', model: 'poly5', mapStep: 4 });
  const seg = r.perSegment[0];
  const view = d.segments[0].views[0];
  const { image, map, window } = resampleView(view, seg);
  assert.ok(image.width > 0 && image.height > 0);
  assert.equal(map.step, 1);
  assert.equal(window.width, image.width);
  // The corrected raster must be mostly real pixels, not transparency.
  let blank = 0;
  for (let i = 3; i < image.data.length; i += 4) if (image.data[i] === 0) blank++;
  const frac = blank / (image.data.length / 4);
  assert.ok(frac < 0.2, `blank fraction ${frac.toFixed(3)} is a thin border at most`);
});
