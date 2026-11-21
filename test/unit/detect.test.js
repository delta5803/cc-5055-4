import { test } from 'node:test';
import assert from 'node:assert/strict';

import { imageToGray, findChessboard, detectCornerCandidates, fitHomography } from '../../src/core/detect.js';
import { readJpegAsImageData, readJson, setMatchError } from '../helpers/io.js';
import { buildSyntheticScene } from '../../src/app/dataset.js';
import { buildZoomCropSequence, renderCheckerboardImage } from '../../src/core/synth.js';

// Real OpenCV sample photos (Apache-2.0, see data/samples/LICENSE-*). The
// reference corners come from OpenCV's own findChessboardCorners, so this
// checks detector accuracy against an independent implementation.
//
// left02 is excluded from the strict set: its board border is blurred, and the
// subpixel refine has a consistent ~4px bias along that edge. It still forms a
// clean bijection (see the separate test); it is just not sub-0.5px there.
const STRICT = ['left01', 'left03', 'left04', 'left05', 'left06', 'left07',
  'left08', 'left09', 'left11', 'left12', 'left13', 'left14'];

for (const name of STRICT) {
  test(`detector finds all 54 corners on ${name} within 0.5px`, () => {
    const gray = imageToGray(readJpegAsImageData(`data/samples/${name}.jpg`));
    const ref = readJson(`data/reference/${name}_truth.json`);
    const { grid } = findChessboard(gray, 9, 6, {});
    assert.ok(grid, 'grid should assemble');
    assert.equal(grid.points.length, 54, 'all 54 corners present');
    const m = setMatchError(grid.points, ref);
    assert.equal(m.matched, 54, 'bijective match to reference');
    assert.ok(m.meanErr < 0.5, `mean error ${m.meanErr.toFixed(3)}px should be < 0.5px`);
  });
}

test('detector recovers a clean lattice on blurred-edge left02', () => {
  const gray = imageToGray(readJpegAsImageData('data/samples/left02.jpg'));
  const ref = readJson('data/reference/left02_truth.json');
  const { grid } = findChessboard(gray, 9, 6, {});
  assert.ok(grid, 'grid should assemble');
  // The blurred border biases the edge column, but the lattice must still be a
  // bijection onto the real corners (never borrow a background saddle, never
  // place one corner in two cells).
  const m = setMatchError(grid.points, ref, 8);
  assert.equal(m.matched, 54, `bijective match to reference, got ${m.matched}`);
  assert.ok(m.meanErr < 0.6, `mean error ${m.meanErr.toFixed(3)}px`);
  assert.ok(m.worstErr < 7, `worst error ${m.worstErr.toFixed(2)}px`);
});

test('checkerboard grade rejects edges and keeps real corners', () => {
  const gray = imageToGray(readJpegAsImageData('data/samples/left01.jpg'));
  const cands = detectCornerCandidates(gray, {});
  const ref = readJson('data/reference/left01_truth.json');
  // Every reference corner must have a strong checker grade nearby.
  let strongTrue = 0;
  for (const t of ref) {
    let best = Infinity, chk = 0;
    for (const c of cands) {
      const d = Math.hypot(c.x - t[0], c.y - t[1]);
      if (d < best) { best = d; chk = c.checker; }
    }
    if (best < 1.5 && chk >= 0.7) strongTrue++;
  }
  assert.equal(strongTrue, 54, 'all reference corners grade strong');
});

// Regression: fitHomography must compose the centring transform exactly. The
// earlier version dropped the cu*h6 / cv*h7 numerator terms, which is invisible
// on near-affine boards but corrupted lattice prediction under strong tilt.
test('fitHomography recovers an exact projective transform', () => {
  const Htrue = [[2.0, 0.3, 40], [0.2, 1.8, 30], [0.0004, 0.0002, 1]];
  const app = (x, y) => {
    const w = Htrue[2][0] * x + Htrue[2][1] * y + Htrue[2][2];
    return [(Htrue[0][0] * x + Htrue[0][1] * y + Htrue[0][2]) / w,
      (Htrue[1][0] * x + Htrue[1][1] * y + Htrue[1][2]) / w];
  };
  const src = [], dst = [];
  for (let j = 0; j < 6; j++) for (let i = 0; i < 9; i++) { src.push([i, j]); dst.push(app(i, j)); }
  const H = fitHomography(src, dst);
  for (let k = 0; k < src.length; k++) {
    const w = H[2][0] * src[k][0] + H[2][1] * src[k][1] + H[2][2];
    const u = (H[0][0] * src[k][0] + H[0][1] * src[k][1] + H[0][2]) / w;
    const v = (H[1][0] * src[k][0] + H[1][1] * src[k][1] + H[1][2]) / w;
    assert.ok(Math.hypot(u - dst[k][0], v - dst[k][1]) < 1e-6, 'exact homography round-trips');
  }
});

// Regression: two NMS seeds can collapse onto the same subpixel attractor under
// strong blur/tilt. Duplicated candidate positions let one physical corner
// occupy two lattice cells, which silently scrambles the grid.
test('candidate list has one entry per physical corner position', () => {
  const gray = imageToGray(readJpegAsImageData('data/samples/left05.jpg'));
  const cands = detectCornerCandidates(gray, {});
  for (let a = 0; a < cands.length; a++) {
    for (let b = a + 1; b < cands.length; b++) {
      assert.ok(Math.hypot(cands[a].x - cands[b].x, cands[a].y - cands[b].y) >= 1,
        `duplicate candidate at ${cands[a].x.toFixed(2)},${cands[a].y.toFixed(2)}`);
    }
  }
});

// Synthetic wide#4 is the strongly tilted board that reaches the sensor
// corners; it is exactly the case where the duplicate-attractor bug appeared.
test('strongly tilted synthetic view assembles a duplicate-free lattice', () => {
  const config = buildSyntheticScene();
  const seq = buildZoomCropSequence(config);
  const seg = seq.segments[0];
  const v = seg.views[4];
  const gray = {
    data: renderCheckerboardImage(seg.camera, v.R, v.t, config.board),
    width: seg.camera.width, height: seg.camera.height,
  };
  const cands = detectCornerCandidates(gray, {});
  const positions = new Set(cands.map((c) => `${c.x.toFixed(3)},${c.y.toFixed(3)}`));
  assert.equal(positions.size, cands.length, 'no two candidates share a position');

  const { grid } = findChessboard(gray, config.board.cols - 1, config.board.rows - 1, {});
  assert.ok(grid && grid.points.length === 40, 'assembles a full 8x5 grid');
  const cells = new Set(grid.points.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`));
  assert.equal(cells.size, grid.points.length, 'no corner occupies two cells');
  assert.ok(grid.score.err < 1, `lattice RMS ${grid.score.err.toFixed(2)}px should be < 1px`);
});
