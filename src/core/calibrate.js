// Camera calibration from planar chessboard views.
//
// Two independent paths, both using the same physical-sensor model:
//
//  A. solvePerFocalLength(views)  -> classic independent solve: one K (pixels)
//     and one distortion polynomial (sensor mm) per image group. This is what
//     you get when you treat every focal/crop setting as its own camera.
//
//  B. solveSharedSensor(segments) -> ONE distortion polynomial in sensor-mm
//     and ONE physical principal point shared across all focal/crop settings,
//     with a separate focal length per segment. This is the "one lens, many
//     zoom settings" model.
//
// Distortion always lives in sensor-mm coordinates in both paths, so the
// resulting coefficients are directly comparable and focal-length independent
// by construction. Pixel-space coefficients for display are derived by
// k_px_i = k_mm_i * f_mm^(2i), p_px = p_mm * f_mm.

import {
  zeros, dot, cross, normalize3, leastSquares, jacobiEigen,
  levenbergMarquardt, parameterCovariance,
} from './linalg.js';
import {
  intrinsicsToSensor, sensorToPixelIntrinsics, distortNormalized,
  projectPoint, normalizedToPixel, sensorXFromPixel, sensorYFromPixel,
  distortJacobianNormalized, jacobianDet, undistortNormalized,
  pixelToIdealNormalized,
} from './lens.js';

// ---------------------------------------------------------------------------
// Homography + pose
// ---------------------------------------------------------------------------

export function homography(objPts, imgPts) {
  const n = objPts.length;
  const cxO = mean(objPts.map((p) => p[0]));
  const cyO = mean(objPts.map((p) => p[1]));
  let sO = 0;
  for (const p of objPts) sO += Math.hypot(p[0] - cxO, p[1] - cyO);
  sO = Math.SQRT2 / (sO / n || 1);
  const cxI = mean(imgPts.map((p) => p[0]));
  const cyI = mean(imgPts.map((p) => p[1]));
  let sI = 0;
  for (const p of imgPts) sI += Math.hypot(p[0] - cxI, p[1] - cyI);
  sI = Math.SQRT2 / (sI / n || 1);

  const To = [[sO, 0, -sO * cxO], [0, sO, -sO * cyO], [0, 0, 1]];
  const Ti = [[sI, 0, -sI * cxI], [0, sI, -sI * cyI], [0, 0, 1]];
  const A = [];
  for (let i = 0; i < n; i++) {
    const x = (objPts[i][0] - cxO) * sO;
    const y = (objPts[i][1] - cyO) * sO;
    const u = (imgPts[i][0] - cxI) * sI;
    const v = (imgPts[i][1] - cyI) * sI;
    A.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    A.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }
  const At = transpose2(A);
  const AtA = mul2(At, A);
  const { values, vectors } = jacobiEigen(AtA);
  let mi = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[mi]) mi = i;
  const h = Array.from({ length: 9 }, (_, r) => vectors[r][mi]);
  const Hn = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];
  return matMul3(matMul3(invert3(Ti), Hn), To);
}

function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }
function transpose2(A) {
  const n = A.length, m = A[0].length;
  const T = Array.from({ length: m }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}
function mul2(A, B) {
  const n = A.length, k = A[0].length, m = B[0].length;
  const C = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) {
    const a = A[i][p];
    if (a === 0) continue;
    for (let j = 0; j < m; j++) C[i][j] += a * B[p][j];
  }
  return C;
}
function matMul3(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
function invert3(A) {
  const [a, b, c] = A[0], [d, e, f] = A[1], [g, h, i] = A[2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!(Math.abs(det) > 1e-14)) throw new Error('singular 3x3');
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

// Zhang closed form for the image of the absolute conic.
export function initialIntrinsics(homographies) {
  const A = [];
  for (const H of homographies) {
    const v = (i, j) => {
      const h = (r, c) => H[r][c];
      return [
        h(0, i) * h(0, j),
        h(0, i) * h(1, j) + h(1, i) * h(0, j),
        h(1, i) * h(1, j),
        h(2, i) * h(0, j) + h(0, i) * h(2, j),
        h(2, i) * h(1, j) + h(1, i) * h(2, j),
        h(2, i) * h(2, j),
      ];
    };
    const v12 = v(0, 1), v11 = v(0, 0), v22 = v(1, 1);
    A.push(v12);
    A.push(v11.map((q, i) => q - v22[i]));
  }
  const At = transpose2(A);
  const AtA = mul2(At, A);
  const { values, vectors } = jacobiEigen(AtA);
  let mi = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[mi]) mi = i;
  const w = Array.from({ length: 6 }, (_, r) => vectors[r][mi]);
  const [B11, B12, B22, B13, B23, B33] = w;
  const d = B11 * B22 - B12 * B12;
  if (!(Math.abs(d) > 1e-18) || Math.abs(B11) < 1e-18) return null;
  const v0 = (B12 * B13 - B11 * B23) / d;
  const lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
  if (!(lambda / B11 > 0)) return null;
  const alpha = Math.sqrt(lambda / B11);
  const beta = Math.sqrt((lambda * B11) / d);
  const gamma = (-B12 * alpha * alpha * beta) / lambda;
  const u0 = (gamma * v0) / beta - (B13 * alpha * alpha) / lambda;
  return { fx: alpha, fy: beta, cx: u0, cy: v0 };
}

export function rodriguesToRotation(r) {
  const theta = Math.hypot(r[0], r[1], r[2]);
  if (theta < 1e-14) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const kx = r[0] / theta, ky = r[1] / theta, kz = r[2] / theta;
  const c = Math.cos(theta), s = Math.sin(theta), C = 1 - c;
  return [
    [c + kx * kx * C, kx * ky * C - kz * s, kx * kz * C + ky * s],
    [ky * kx * C + kz * s, c + ky * ky * C, ky * kz * C - kx * s],
    [kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C],
  ];
}

export function rotationToRodrigues(R) {
  const tr = R[0][0] + R[1][1] + R[2][2];
  const cosT = Math.min(1, Math.max(-1, (tr - 1) / 2));
  const theta = Math.acos(cosT);
  if (theta < 1e-12) return [0, 0, 0];
  if (Math.abs(theta - Math.PI) < 1e-5) {
    const xx = Math.sqrt(Math.max(0, (R[0][0] + 1) / 2));
    const yy = Math.sqrt(Math.max(0, (R[1][1] + 1) / 2));
    const zz = Math.sqrt(Math.max(0, (R[2][2] + 1) / 2));
    let x = xx, y = yy, z = zz;
    if (R[0][1] < 0) y = -y;
    if (R[0][2] < 0) z = -z;
    const n = Math.hypot(x, y, z) || 1;
    return [(x / n) * theta, (y / n) * theta, (z / n) * theta];
  }
  const f = theta / (2 * Math.sin(theta));
  return [f * (R[2][1] - R[1][2]), f * (R[0][2] - R[2][0]), f * (R[1][0] - R[0][1])];
}

export function poseFromHomography(H, K) {
  const M = matMul3(invert3(K), H);
  const n1 = Math.hypot(M[0][0], M[1][0], M[2][0]);
  const n2 = Math.hypot(M[0][1], M[1][1], M[2][1]);
  const lambda = 2 / (n1 + n2);
  let r1 = [M[0][0] * lambda, M[1][0] * lambda, M[2][0] * lambda];
  let r2 = [M[0][1] * lambda, M[1][1] * lambda, M[2][1] * lambda];
  let t = [M[0][2] * lambda, M[1][2] * lambda, M[2][2] * lambda];
  if (t[2] < 0) { r1 = r1.map((v) => -v); r2 = r2.map((v) => -v); t = t.map((v) => -v); }
  r1 = normalize3(r1);
  const proj = dot(r1, r2);
  r2 = normalize3(r2.map((v, i) => v - proj * r1[i]));
  const r3 = cross(r1, r2);
  return {
    R: [[r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]]],
    t,
  };
}

// ---------------------------------------------------------------------------
// Path A: independent per-image-group calibration
// ---------------------------------------------------------------------------

// views: [{ objectPoints, imagePoints, width, height }]
// opts: { sensor:{widthMM,heightMM,originX_mm,originY_mm}, model, fix..., imageSize }
export function solvePerFocalLength(views, opts = {}) {
  // Same conditioning issue as solveSharedSensor: solve the 4-term model
  // first and use it to seed the 5-term fit. Keep the 5-term result only if it
  // improves the residual without a runaway focal length.
  const model = opts.model ?? 'poly5';
  const poly2 = solvePerFocalLengthModel(views, { ...opts, model: 'poly2' });
  if (model !== 'poly5') return poly2;
  const poly5 = solvePerFocalLengthModel(views, { ...opts, model: 'poly5' }, {
    fx: poly2.K.fx, fy: poly2.K.fy, cx: poly2.K.cx, cy: poly2.K.cy,
    distortion: poly2.distortionMM,
  });
  const focalSane = poly5.K.fx > 0.5 * poly2.K.fx && poly5.K.fx < 1.5 * poly2.K.fx;
  if (poly5.rms < poly2.rms * (1 - 1e-6) && focalSane) return poly5;
  return { ...poly2, model: 'poly2', note: poly5.rms < poly2.rms ? 'poly5 rejected: implausible focal length' : 'poly5 gave no residual improvement' };
}

function solvePerFocalLengthModel(views, opts = {}, init = null) {
  if (views.length < 2) throw new Error('need at least 2 views');
  const width = opts.imageSize?.width ?? views[0].width;
  const height = opts.imageSize?.height ?? views[0].height;
  const useK3 = (opts.model ?? 'poly5') === 'poly5';
  const sensor = opts.sensor ?? {};
  const originX_mm = sensor.originX_mm ?? 0;
  const originY_mm = sensor.originY_mm ?? 0;

  const Hs = views.map((v) => homography(v.objectPoints, v.imagePoints));
  let K0 = initialIntrinsics(Hs) ?? {};
  const fguess = Math.max(width, height);
  if (init) {
    K0 = { fx: init.fx, fy: init.fy, cx: init.cx, cy: init.cy };
  }
  if (!(K0.fx > 0.2 * width && K0.fx < 10 * width)) K0.fx = fguess;
  if (!(K0.fy > 0.2 * height && K0.fy < 10 * height)) K0.fy = K0.fx;
  const maxOff = 0.35 * Math.min(width, height);
  const ccx = (width - 1) / 2, ccy = (height - 1) / 2;
  if (!(Number.isFinite(K0.cx)) || Math.abs(K0.cx - ccx) > maxOff) K0.cx = ccx;
  if (!(Number.isFinite(K0.cy)) || Math.abs(K0.cy - ccy) > maxOff) K0.cy = ccy;
  K0.fx = Math.min(10 * width, Math.max(0.2 * width, K0.fx));
  K0.fy = Math.min(10 * height, Math.max(0.2 * height, K0.fy));

  const poses = Hs.map((H) => poseFromHomography(H, [[K0.fx, 0, K0.cx], [0, K0.fy, K0.cy], [0, 0, 1]]));

  // Parameters: [fx, fy, cx, cy, k1,k2,p1,p2(,k3) | per-view rvec,tvec]
  const d0 = init?.distortion ?? { k1: 0, k2: 0, p1: 0, p2: 0, k3: 0 };
  const P = [K0.fx, K0.fy, K0.cx, K0.cy, d0.k1, d0.k2, d0.p1, d0.p2];
  if (useK3) P.push(d0.k3 ?? 0);
  const nK = useK3 ? 9 : 8;
  const poseOffset = P.length;
  for (const p of poses) {
    const rv = rotationToRodrigues(p.R);
    P.push(rv[0], rv[1], rv[2], p.t[0], p.t[1], p.t[2]);
  }
  const free = new Array(P.length).fill(true);
  if (opts.fixPrincipalPoint) { free[2] = false; free[3] = false; }
  if (opts.fixFocal) { free[0] = false; free[1] = false; }
  if (opts.fixDistortion) for (let i = 4; i < nK; i++) free[i] = false;

  const sensorFor = (fx, fy, cx, cy) => intrinsicsToSensor(
    { fx, fy, cx, cy, width, height },
    { ...sensor, originX_mm, originY_mm },
  );

  const residualFn = (x) => {
    const s = sensorFor(x[0], x[1], x[2], x[3]);
    const D = { k1: x[4], k2: x[5], p1: x[6], p2: x[7], k3: useK3 ? x[8] : 0 };
    const res = [];
    for (let i = 0; i < views.length; i++) {
      const b = poseOffset + i * 6;
      const R = rodriguesToRotation([x[b], x[b + 1], x[b + 2]]);
      const t = [x[b + 3], x[b + 4], x[b + 5]];
      const v = views[i];
      for (let k = 0; k < v.objectPoints.length; k++) {
        const [X, Y, Z] = v.objectPoints[k];
        const Xc = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0];
        const Yc = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1];
        const Zc = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2];
        if (Zc <= 1e-9) { res.push(1e4, 1e4); continue; }
        const pr = projectPoint(Xc, Yc, Zc, s, D);
        res.push(pr.u - v.imagePoints[k][0], pr.v - v.imagePoints[k][1]);
      }
    }
    return Float64Array.from(res);
  };

  const step = new Float64Array(P.length);
  step[0] = step[1] = Math.max(1e-5 * P[0], 1e-6);
  step[2] = step[3] = 1e-5;
  step[4] = step[5] = 1e-10; step[6] = step[7] = 1e-12; if (useK3) step[8] = 1e-15;
  for (let i = poseOffset; i < P.length; i++) step[i] = 1e-7;

  const fit = levenbergMarquardt(residualFn, P, {
    free, step, maxIter: opts.maxIter ?? 200, lambda0: 1e-3, tol: 1e-12,
  });
  const x = fit.x;
  const D = { k1: x[4], k2: x[5], p1: x[6], p2: x[7], k3: useK3 ? x[8] : 0 };
  const K = { fx: x[0], fy: x[1], cx: x[2], cy: x[3], width, height };
  const s = sensorFor(x[0], x[1], x[2], x[3]);

  let sumSq = 0, count = 0;
  const perView = [];
  for (let i = 0; i < views.length; i++) {
    const b = poseOffset + i * 6;
    const R = rodriguesToRotation([x[b], x[b + 1], x[b + 2]]);
    const t = [x[b + 3], x[b + 4], x[b + 5]];
    let sq = 0;
    const rp = [];
    for (let k = 0; k < views[i].objectPoints.length; k++) {
      const [X, Y, Z] = views[i].objectPoints[k];
      const Xc = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0];
      const Yc = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1];
      const Zc = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2];
      const pr = projectPoint(Xc, Yc, Zc, s, D);
      const du = pr.u - views[i].imagePoints[k][0];
      const dv = pr.v - views[i].imagePoints[k][1];
      sq += du * du + dv * dv;
      rp.push([pr.u, pr.v]);
    }
    perView.push({ rms: Math.sqrt(sq / views[i].objectPoints.length), R, t, reprojected: rp });
    sumSq += sq; count += views[i].objectPoints.length;
  }
  const rms = Math.sqrt(sumSq / count);
  const dof = Math.max(1, 2 * count - free.filter(Boolean).length);
  const cov = fit.jacobian && free.slice(0, nK).every(Boolean)
    ? parameterCovariance(fit.jacobian, fit.freeIdx.slice(0, nK), sumSq / dof) : null;
  const std = cov ? Array.from({ length: nK }, (_, i) => Math.sqrt(Math.max(0, cov[i][i]))) : null;

  return {
    K, sensor: s, distortionMM: D, rms, perView,
    poses: perView.map((p) => ({ R: p.R, t: p.t })),
    converged: fit.converged, iterations: fit.iterations,
    std, sensorMM: { widthMM: s.widthMM, heightMM: s.heightMM, originX_mm, originY_mm },
    model: opts.model ?? 'poly5',
  };
}

// Convert a sensor-mm distortion to pixel-normalised coefficients for a focal
// length f_mm: k_px_i = k_mm_i * f_mm^(2i), p_px = p_mm * f_mm.
export function distortionMMToPixel(D, f_mm) {
  return {
    k1: D.k1 * f_mm ** 2,
    k2: D.k2 * f_mm ** 4,
    k3: D.k3 * f_mm ** 6,
    p1: D.p1 * f_mm,
    p2: D.p2 * f_mm,
  };
}

export function distortionPixelToMM(D, f_mm) {
  return {
    k1: D.k1 / f_mm ** 2,
    k2: D.k2 / f_mm ** 4,
    k3: D.k3 / f_mm ** 6,
    p1: D.p1 / f_mm,
    p2: D.p2 / f_mm,
  };
}

// ---------------------------------------------------------------------------
// Path B: shared physical sensor across focal/crop segments
// ---------------------------------------------------------------------------

// segments: [{ focalId, originX_mm?, originY_mm?, views:[{objectPoints,
//               imagePoints,width,height}] }]
export function solveSharedSensor(segments, opts = {}) {
  // A direct 5-term (k1,k2,k3,p1,p2) joint fit from zero distortion sometimes
  // lands in a degenerate minimum where one segment's focal length runs away
  // (the k3 term can absorb a focal change). Solve the 4-term model first —
  // it is well conditioned — then use it to initialise the 5-term fit and only
  // keep that result when it genuinely improves the residual without moving
  // any focal length implausibly far.
  const model = opts.model ?? 'poly5';
  const poly2 = solveSharedSensorModel(segments, { ...opts, model: 'poly2' });
  if (model !== 'poly5') return poly2;
  const poly5 = solveSharedSensorModel(segments, { ...opts, model: 'poly5' }, {
    px_mm: poly2.principalMM.px_mm,
    py_mm: poly2.principalMM.py_mm,
    distortion: poly2.distortionMM,
    focalMM: poly2.segments.map((s) => s.f_mm),
  });
  const fb = poly2.segments.map((s) => s.f_mm);
  const f5 = poly5.segments.map((s) => s.f_mm);
  const focalSane = f5.every((f, i) => f > 0.5 * fb[i] && f < 1.5 * fb[i]);
  const better = poly5.rms < poly2.rms * 1e-6;
  if (better && focalSane) return poly5;
  return { ...poly2, model: 'poly2', note: better ? 'poly5 rejected: implausible focal length' : 'poly5 gave no residual improvement' };
}

function solveSharedSensorModel(segments, opts = {}, init = null) {
  if (segments.length === 0) throw new Error('no segments');
  const width = opts.imageSize?.width ?? segments[0].views[0].width;
  const height = opts.imageSize?.height ?? segments[0].views[0].height;
  const sensor = opts.sensor ?? {};
  const useK3 = (opts.model ?? 'poly5') === 'poly5';
  // Pitch is a sensor property and must NOT be re-derived from a cropped
  // image's pixel count. Callers with cropped segments should pass pitchX/Y.
  const pitchX = sensor.pitchX ?? (sensor.widthMM ?? 36) / width;
  const pitchY = sensor.pitchY ?? (sensor.heightMM ?? (36 * height) / width) / height;

  const perSeg = segments.map((seg, si) => {
    try {
      return solvePerFocalLength(seg.views, {
        ...opts,
        model: 'poly2',
        sensor: {
          ...sensor,
          pitchX, pitchY,
          originX_mm: seg.originX_mm ?? 0,
          originY_mm: seg.originY_mm ?? 0,
        },
      });
    } catch (e) {
      throw new Error(`segment ${si} init failed: ${e.message}`);
    }
  });

  // Parameters: [fx_mm per segment, px_mm, py_mm, k1,k2,p1,p2(,k3) | poses]
  const nSeg = segments.length;
  const P = [];
  const segFocalIdx = [];
  for (let si = 0; si < nSeg; si++) {
    segFocalIdx.push(P.length);
    P.push(init?.focalMM?.[si] ?? perSeg[si].sensor.fx_mm);
  }
  const pxIdx = P.length; P.push(init?.px_mm ?? perSeg[0].sensor.px_mm);
  const pyIdx = P.length; P.push(init?.py_mm ?? perSeg[0].sensor.py_mm);
  const dIdx = P.length;
  const d0 = init?.distortion ?? { k1: 0, k2: 0, p1: 0, p2: 0, k3: 0 };
  P.push(d0.k1, d0.k2, d0.p1, d0.p2); if (useK3) P.push(d0.k3 ?? 0);
  const nDist = useK3 ? 5 : 4;
  const poseOffset = P.length;
  const poseSlots = [];
  for (let si = 0; si < nSeg; si++) {
    const slots = [];
    for (let vi = 0; vi < segments[si].views.length; vi++) {
      slots.push(P.length);
      const rv = rotationToRodrigues(perSeg[si].poses[vi].R);
      const t = perSeg[si].poses[vi].t;
      P.push(rv[0], rv[1], rv[2], t[0], t[1], t[2]);
    }
    poseSlots.push(slots);
  }

  const segOrigin = (si) => ({
    x: segments[si].originX_mm ?? 0,
    y: segments[si].originY_mm ?? 0,
  });

  const residualFn = (x) => {
    const px_mm = x[pxIdx], py_mm = x[pyIdx];
    const D = { k1: x[dIdx], k2: x[dIdx + 1], p1: x[dIdx + 2], p2: x[dIdx + 3], k3: useK3 ? x[dIdx + 4] : 0 };
    const res = [];
    for (let si = 0; si < nSeg; si++) {
      const f_mm = x[segFocalIdx[si]];
      const o = segOrigin(si);
      for (let vi = 0; vi < segments[si].views.length; vi++) {
        const b = poseSlots[si][vi];
        const R = rodriguesToRotation([x[b], x[b + 1], x[b + 2]]);
        const t = [x[b + 3], x[b + 4], x[b + 5]];
        const view = segments[si].views[vi];
        const vw = segments[si].views[vi].width ?? width;
        const vh = segments[si].views[vi].height ?? height;
        for (let k = 0; k < view.objectPoints.length; k++) {
          const [X, Y, Z] = view.objectPoints[k];
          const Xc = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0];
          const Yc = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1];
          const Zc = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2];
          if (Zc <= 1e-9) { res.push(1e4, 1e4); continue; }
          const d = distortNormalized((f_mm * Xc) / Zc, (f_mm * Yc) / Zc, D);
          const sx = px_mm + d.xd;
          const sy = py_mm + d.yd;
          const u = (sx - o.x) / pitchX + (vw - 1) / 2;
          const v = (sy - o.y) / pitchY + (vh - 1) / 2;
          res.push(u - view.imagePoints[k][0], v - view.imagePoints[k][1]);
        }
      }
    }
    return Float64Array.from(res);
  };

  const free = new Array(P.length).fill(true);
  if (opts.fixPrincipalPoint) { free[pxIdx] = false; free[pyIdx] = false; }
  if (opts.fixDistortion) for (let i = 0; i < nDist; i++) free[dIdx + i] = false;
  const step = new Float64Array(P.length);
  for (let si = 0; si < nSeg; si++) step[segFocalIdx[si]] = 1e-6;
  step[pxIdx] = step[pyIdx] = 1e-5;
  step[dIdx] = step[dIdx + 1] = 1e-10;
  step[dIdx + 2] = step[dIdx + 3] = 1e-12;
  if (useK3) step[dIdx + 4] = 1e-15;
  for (let i = poseOffset; i < P.length; i++) step[i] = 1e-7;

  const fit = levenbergMarquardt(residualFn, P, {
    free, step, maxIter: opts.maxIter ?? 300, lambda0: 1e-3, tol: 1e-12,
  });
  const x = fit.x;
  const D = { k1: x[dIdx], k2: x[dIdx + 1], p1: x[dIdx + 2], p2: x[dIdx + 3], k3: useK3 ? x[dIdx + 4] : 0 };
  const px_mm = x[pxIdx], py_mm = x[pyIdx];

  const segmentsOut = [];
  let sumSq = 0, count = 0;
  for (let si = 0; si < nSeg; si++) {
    const f_mm = x[segFocalIdx[si]];
    const o = segOrigin(si);
    const w = segments[si].views[0].width ?? width;
    const h = segments[si].views[0].height ?? height;
    const s = {
      width: w, height: h, pitchX, pitchY,
      fx_mm: f_mm, fy_mm: f_mm, px_mm, py_mm,
      originX_mm: o.x, originY_mm: o.y,
      widthMM: sensor.widthMM ?? 36,
      heightMM: sensor.heightMM ?? (36 * height) / width,
    };
    const viewsOut = [];
    for (let vi = 0; vi < segments[si].views.length; vi++) {
      const b = poseSlots[si][vi];
      const R = rodriguesToRotation([x[b], x[b + 1], x[b + 2]]);
      const t = [x[b + 3], x[b + 4], x[b + 5]];
      const view = segments[si].views[vi];
      let sq = 0;
      const rp = [];
      for (let k = 0; k < view.objectPoints.length; k++) {
        const [X, Y, Z] = view.objectPoints[k];
        const Xc = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0];
        const Yc = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1];
        const Zc = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2];
        const pr = projectPoint(Xc, Yc, Zc, s, D);
        const du = pr.u - view.imagePoints[k][0];
        const dv = pr.v - view.imagePoints[k][1];
        sq += du * du + dv * dv;
        rp.push([pr.u, pr.v]);
      }
      viewsOut.push({ rms: Math.sqrt(sq / view.objectPoints.length), R, t, reprojected: rp });
      sumSq += sq; count += view.objectPoints.length;
    }
    segmentsOut.push({
      focalId: segments[si].focalId, f_mm,
      K: sensorToPixelIntrinsics(s), sensor: s, views: viewsOut,
    });
  }
  return {
    distortionMM: D,
    principalMM: { px_mm, py_mm },
    segments: segmentsOut,
    rms: Math.sqrt(sumSq / count),
    converged: fit.converged,
    iterations: fit.iterations,
    model: opts.model ?? 'poly5',
    sensorShared: {
      widthMM: sensor.widthMM ?? 36,
      heightMM: sensor.heightMM ?? (36 * height) / width,
      pitchX, pitchY,
    },
    width, height,
  };
}

export { distortJacobianNormalized, jacobianDet, normalizedToPixel, sensorXFromPixel, sensorYFromPixel };

// ---------------------------------------------------------------------------
// Model selection and coverage diagnostics
// ---------------------------------------------------------------------------

// Akaike information criterion for choosing between poly2 and poly5 by
// residual cost, with the number of free intrinsic+distortion parameters.
export function modelAIC(fit, nObs) {
  const k = fit.std ? fit.std.length : 0;
  const cost = fit.rms * fit.rms * nObs;
  return nObs * Math.log(Math.max(cost / nObs, 1e-30)) + 2 * k;
}

// How far from the optical axis (as a fraction of the image half-diagonal) do
// the calibration corners actually reach? This is what decides whether edge
// undistortion is extrapolation.
export function coverageStats(views, K) {
  const width = K.width, height = K.height;
  const cxp = K.cx, cyp = K.cy;
  const halfDiag = Math.hypot(width, height) / 2;
  let maxR = 0;
  let sumR = 0, n = 0;
  const byQuadrant = [0, 0, 0, 0];
  for (const v of views) {
    for (const [u, w] of v.imagePoints) {
      const r = Math.hypot(u - cxp, w - cyp) / halfDiag;
      maxR = Math.max(maxR, r);
      sumR += r; n++;
      const q = (u >= cxp ? 1 : 0) + (w >= cyp ? 2 : 0);
      byQuadrant[q]++;
    }
  }
  return { maxRadialFraction: maxR, meanRadialFraction: sumR / n, byQuadrant };
}

// Evaluate a fitted model on a rectangular grid of *ideal* output pixels and
// report where the inverse map fails (fold / out-of-coverage). Used to compare
// shared vs per-focal in the image periphery.
export function edgeBehaviour(s, k, width, height, grid = 33) {
  const samples = [];
  let fails = 0, folds = 0, outside = 0, maxRoundTrip = 0;
  for (let j = 0; j < grid; j++) {
    for (let i = 0; i < grid; i++) {
      const u = (i / (grid - 1)) * (width - 1);
      const v = (j / (grid - 1)) * (height - 1);
      // Inverse solve in sensor-mm (coefficient units).
      const inv = pixelToIdealNormalized(u, v, s, k);
      // Forward map from the same ideal pixel: destination -> source.
      const fwd = normalizedToPixel(inv.x, inv.y, s, k);
      const rt = roundTrip(u, v, s, k);
      maxRoundTrip = Math.max(maxRoundTrip, rt.errorPx);
      let status = 'ok';
      if (!inv.converged || inv.jacobianDet <= 0) { status = 'fold'; folds++; fails++; }
      else if (fwd.u < -0.5 || fwd.v < -0.5 || fwd.u > s.width - 0.5 || fwd.v > s.height - 0.5) { status = 'outside'; outside++; fails++; }
      else if (rt.errorPx > 0.05) { status = 'unstable'; fails++; }
      samples.push({ u, v, status, errorPx: rt.errorPx });
    }
  }
  return {
    samples, fails, folds, outside,
    failureFraction: fails / samples.length,
    maxRoundTripPx: maxRoundTrip,
    edgeFailure: samples.filter((q) => {
      const dx = (q.u - width / 2) / (width / 2), dy = (q.v - height / 2) / (height / 2);
      return Math.max(Math.abs(dx), Math.abs(dy)) > 0.75 && q.status !== 'ok';
    }).length,
  };
}

function undistortNormalizedSafe(x, y, k) {
  try { return undistortNormalized(x, y, k); } catch { return null; }
}

function roundTrip(u, v, s, k) {
  const inv = pixelToIdealNormalized(u, v, s, k);
  if (!inv.converged) return { errorPx: Infinity };
  const pr = normalizedToPixel(inv.x, inv.y, s, k); // forward: ideal -> observed
  // Re-invert the observed point; it must land back on the ideal pixel.
  const back = pixelToIdealNormalized(pr.u, pr.v, s, k);
  if (!back.converged) return { errorPx: Infinity };
  const b = normalizedToPixel(back.x, back.y, s, k);
  return { errorPx: Math.hypot(b.u - u, b.v - v) };
}

