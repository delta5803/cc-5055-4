import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  homography, initialIntrinsics, poseFromHomography, rodriguesToRotation,
  rotationToRodrigues, solvePerFocalLength, solveSharedSensor,
  distortionMMToPixel, distortionPixelToMM,
} from '../../src/core/calibrate.js';
import { buildSyntheticScene } from '../../src/app/dataset.js';
import {
  buildZoomCropSequence, renderCheckerboardImage, renderView,
  checkerboardObjectPoints, makeCamera, eulerRotation,
} from '../../src/core/synth.js';
import { findChessboard } from '../../src/core/detect.js';

function applyH(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

test('homography recovers an exact projective mapping', () => {
  const Ht = [[1.7, 0.2, 13], [-0.1, 1.6, -7], [3e-4, -2e-4, 1]];
  const obj = [], img = [];
  for (let j = 0; j < 5; j++) for (let i = 0; i < 8; i++) { obj.push([i, j]); img.push(applyH(Ht, i, j)); }
  const H = homography(obj, img);
  for (let k = 0; k < obj.length; k++) {
    const p = applyH(H, obj[k][0], obj[k][1]);
    assert.ok(Math.hypot(p[0] - img[k][0], p[1] - img[k][1]) < 1e-8);
  }
});

test('initialIntrinsics recovers focal and principal point from clean homographies', () => {
  // Build homographies from a known camera, then check Zhang recovers it.
  const W = 800, Hh = 600;
  const cam = makeCamera({ width: W, height: Hh, sensorMM: 36, fx_mm: 44, fy_mm: 44, px_mm: 0.7, py_mm: -0.4 });
  const board = { cols: 9, rows: 6, squareSize: 20 };
  board.objectPoints = checkerboardObjectPoints(board.cols, board.rows, board.squareSize);
  const Hs = [];
  const poses = [
    { rx: 5, ry: 0, rz: 0, t: [0, 0, 600] },
    { rx: -8, ry: 6, rz: 4, t: [-30, 10, 650] },
    { rx: 10, ry: -7, rz: -5, t: [25, -15, 700] },
  ];
  for (const pose of poses) {
    const view = renderView(cam, board, { R: eulerRotation(pose.rx, pose.ry, pose.rz), t: pose.t });
    // object point (X,Y,0) -> image pixel
    const obj = [], img = [];
    for (let k = 0; k < view.imagePoints.length; k++) {
      if (!view.visible[k]) continue;
      obj.push([view.objectPoints[k][0], view.objectPoints[k][1]]);
      img.push(view.imagePoints[k]);
    }
    if (obj.length >= 4) Hs.push(homography(obj, img));
  }
  assert.ok(Hs.length >= 2);
  const K = initialIntrinsics(Hs);
  assert.ok(K, 'returns intrinsics');
  assert.ok(Math.abs(K.fx - 44 / (36 / W)) < 0.03 * (44 / (36 / W)), `fx near truth, got ${K.fx}`);
  assert.ok(Math.abs(K.fy - 44 / (36 / W)) < 0.03 * (44 / (36 / W)), `fy near truth, got ${K.fy}`);
});

test('poseFromHomography round-trips a known rotation', () => {
  const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
  const R = rodriguesToRotation([0.1, -0.2, 0.3]);
  const t = [-20, 15, 500];
  // Assemble H = K [r1 r2 t]
  const M = [
    [R[0][0], R[0][1], t[0]],
    [R[1][0], R[1][1], t[1]],
    [R[2][0], R[2][1], t[2]],
  ];
  const H = matMul3Test(K, M);
  const pose = poseFromHomography(H, K);
  // Compare rotations via trace of R^T Rtrue
  let tr = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) tr += pose.R[i][j] * R[i][j];
  const cosang = Math.min(1, Math.max(-1, (tr - 1) / 2));
  assert.ok(Math.acos(cosang) < 1e-3, 'rotation recovered');
  assert.ok(Math.hypot(pose.t[0] - t[0], pose.t[1] - t[1], pose.t[2] - t[2]) < 1e-2 * 500);
});

function matMul3Test(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}

test('rodrigues rotation round-trips', () => {
  for (const r of [[0.1, 0.2, -0.3], [0, 0, 0], [1.2, -0.5, 0.9]]) {
    const R = rodriguesToRotation(r);
    const back = rotationToRodrigues(R);
    assert.ok(Math.hypot(back[0] - r[0], back[1] - r[1], back[2] - r[2]) < 1e-9);
  }
});

test('distortionMMToPixel and distortionPixelToMM are inverses', () => {
  const D = { k1: -3e-4, k2: 1.2e-7, k3: 0, p1: 3.4e-5, p2: -1.2e-5 };
  const f = 30;
  const P = distortionMMToPixel(D, f);
  const back = distortionPixelToMM(P, f);
  for (const key of ['k1', 'k2', 'k3', 'p1', 'p2']) {
    assert.ok(Math.abs(back[key] - D[key]) < 1e-15 * Math.max(1, Math.abs(D[key])));
  }
});

// End-to-end: build the synthetic zoom+crop scene, detect, solve both paths,
// and require the truth camera to be recovered to sub-pixel reprojection.
test('per-focal and shared solves recover the synthetic truth', () => {
  const config = buildSyntheticScene();
  const seq = buildZoomCropSequence(config);
  const segments = seq.segments.map((seg) => {
    const views = seg.views.map((v) => {
      const grayData = renderCheckerboardImage(seg.camera, v.R, v.t, config.board);
      const { grid } = findChessboard(
        { data: grayData, width: seg.camera.width, height: seg.camera.height },
        config.board.cols - 1, config.board.rows - 1, {});
      const objectPoints = [], imagePoints = [];
      for (let k = 0; k < grid.points.length; k++) {
        const [i, j] = grid.gridIJ[k];
        objectPoints.push([i * config.board.squareSize, j * config.board.squareSize, 0]);
        imagePoints.push(grid.points[k]);
      }
      return { objectPoints, imagePoints, width: seg.camera.width, height: seg.camera.height };
    });
    return { focalId: seg.focalId, views, width: seg.camera.width, height: seg.camera.height,
      originX_mm: seg.crop?.originX_mm ?? 0, originY_mm: seg.crop?.originY_mm ?? 0 };
  });
  const sensor = { widthMM: 36, heightMM: 27, pitchX: config.pitch ?? 36 / 640, pitchY: 36 / 640 };

  // Per-focal wide segment.
  const wide = segments[0];
  const per = solvePerFocalLength(wide.views, { sensor, model: 'poly2' });
  assert.ok(per.rms < 1, `per-focal reprojection ${per.rms.toFixed(3)} px`);
  assert.ok(Math.abs(per.sensor.fx_mm - 30) < 1.5, `focal near 30mm, got ${per.sensor.fx_mm}`);

  // Shared sensor across all three segments.
  const shared = solveSharedSensor(segments.map((s) => ({
    focalId: s.focalId, originX_mm: s.originX_mm, originY_mm: s.originY_mm, views: s.views,
  })), { sensor: { ...sensor, heightMM: 36 * 480 / 640 }, imageSize: { width: 640, height: 480 }, model: 'poly2' });
  assert.ok(shared.converged || shared.segments.length === 3);
  assert.ok(shared.rms < 1.5, `shared overall rms ${shared.rms.toFixed(3)}`);
  for (const s of shared.segments) {
    const viewRms = s.views.reduce((a, v) => a + v.rms, 0) / Math.max(1, s.views.length);
    assert.ok(viewRms < 1.5, `${s.focalId} shared rms ${viewRms.toFixed(3)}`);
  }
});
