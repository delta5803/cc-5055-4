'use strict';

// Small dense linear-algebra helpers used by the calibration solvers.
// Everything is plain Float64Array / arrays of arrays; sizes are tiny
// (<= a few hundred), so clarity beats micro-optimisation.

function zeros(n, m) {
  if (m === undefined) return new Float64Array(n);
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = new Float64Array(m);
  return a;
}

function cloneMat(A) {
  return A.map((r) => Float64Array.from(r));
}

function transpose(A) {
  const n = A.length;
  const m = A[0].length;
  const T = new Array(m);
  for (let j = 0; j < m; j++) {
    T[j] = new Float64Array(n);
    for (let i = 0; i < n; i++) T[j][i] = A[i][j];
  }
  return T;
}

function matMul(A, B) {
  const n = A.length;
  const k = A[0].length;
  const m = B[0].length;
  const C = new Array(n);
  for (let i = 0; i < n; i++) {
    const Ci = new Float64Array(m);
    const Ai = A[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < m; j++) Ci[j] += a * Bp[j];
    }
    C[i] = Ci;
  }
  return C;
}

function matVec(A, x) {
  const n = A.length;
  const m = A[0].length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const Ai = A[i];
    for (let j = 0; j < m; j++) s += Ai[j] * x[j];
    y[i] = s;
  }
  return y;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize3(a) {
  const n = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / n, a[1] / n, a[2] / n];
}

// Solve A x = b with partial-pivot LU. A is modified (copy if needed).
function solveLU(Ain, bin) {
  const n = Ain.length;
  const A = cloneMat(Ain);
  const b = Float64Array.from(bin);
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let best = Math.abs(A[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(A[r][col]);
      if (v > best) { best = v; pivot = r; }
    }
    if (best < 1e-300) return null;
    if (pivot !== col) {
      const tmp = A[pivot]; A[pivot] = A[col]; A[col] = tmp;
      const ti = perm[pivot]; perm[pivot] = perm[col]; perm[col] = ti;
    }
    const d = A[col][col];
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / d;
      if (f === 0) continue;
      A[r][col] = 0;
      for (let c = col + 1; c < n; c++) A[r][c] -= f * A[col][c];
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[perm[i]];
    for (let j = 0; j < i; j++) s -= A[i][j] * y[j];
    y[i] = s;
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}

// Solve a symmetric positive-definite system via Cholesky with a tiny ridge
// fallback. Returns null when the matrix is numerically singular.
function solveSPD(Ain, bin) {
  const n = Ain.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = Ain[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(s > 0)) return solveLU(Ain, bin);
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = bin[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

function inverse(Ain) {
  const n = Ain.length;
  const cols = [];
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n);
    e[j] = 1;
    const col = solveLU(Ain, e);
    if (!col) return null;
    cols.push(col);
  }
  const out = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i][j] = cols[j][i];
  return out;
}

// Cyclic Jacobi eigensolver for a real symmetric matrix. Returns
// { values: Float64Array, vectors: array-of-arrays } with vectors as columns.
function jacobiEigen(Ain, maxSweeps = 100) {
  const n = Ain.length;
  const A = cloneMat(Ain);
  const V = zeros(n, n);
  for (let i = 0; i < n; i++) V[i][i] = 1;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i][i];
  return { values, vectors: V };
}

// Least-squares via normal equations. A is m x n (n small).
function leastSquares(A, b) {
  const n = A[0].length;
  const At = transpose(A);
  const AtA = matMul(At, A);
  const Atb = matVec(At, b);
  for (let i = 0; i < n; i++) AtA[i][i] += 1e-12;
  return solveSPD(AtA, Atb);
}

// SVD-lite: eigen-decomposition of A^T A. Returns singular values and right
// singular vectors (columns of V), sorted descending.
function svdSmall(A) {
  const At = transpose(A);
  const AtA = matMul(At, A);
  const { values, vectors } = jacobiEigen(AtA);
  const idx = Array.from({ length: values.length }, (_, i) => i).sort((a, b) => values[b] - values[a]);
  const s = idx.map((i) => Math.sqrt(Math.max(0, values[i])));
  const V = zeros(A[0].length, A[0].length);
  idx.forEach((src, dst) => { for (let r = 0; r < V.length; r++) V[r][dst] = vectors[r][src]; });
  return { s, V };
}

// Levenberg-Marquardt in *physical* parameter units.
//   residualFn(x) -> Float64Array of residuals
//   x0            -> physical parameter vector
//   opts.free     -> optional boolean mask; false params stay fixed
//   opts.step     -> per-parameter finite-difference step (defaults to
//                    max(1e-7*|x|, 1e-9))
//   opts.maxIter, opts.lambda0, opts.tol
// Returns { x, residuals, cost, iterations, converged, jacobian, freeIdx }.
// The Jacobian columns refer to the free parameters, in freeIdx order, and
// are derivatives with respect to physical units.
function levenbergMarquardt(residualFn, x0, opts = {}) {
  const free = opts.free || new Array(x0.length).fill(true);
  const n = x0.length;
  const freeIdx = [];
  for (let i = 0; i < n; i++) if (free[i]) freeIdx.push(i);
  const mFree = freeIdx.length;
  let x = Float64Array.from(x0);
  let r = residualFn(x);
  let cost = 0.5 * dot(r, r);
  let lambda = opts.lambda0 ?? 1e-3;
  const maxIter = opts.maxIter ?? 100;
  const tol = opts.tol ?? 1e-12;
  const step = opts.step || new Float64Array(n);
  for (let i = 0; i < n; i++) if (!step[i]) step[i] = Math.max(Math.abs(x[i]) * 1e-7, 1e-9);
  let iter = 0;
  let converged = false;
  let lastJ = null;

  for (; iter < maxIter; iter++) {
    const m = r.length;
    const J = new Array(m);
    for (let i = 0; i < m; i++) J[i] = new Float64Array(mFree);
    for (let c = 0; c < mFree; c++) {
      const j = freeIdx[c];
      const old = x[j];
      const h = step[j];
      x[j] = old + h;
      const rp = residualFn(x);
      x[j] = old;
      for (let i = 0; i < m; i++) J[i][c] = (rp[i] - r[i]) / h;
    }
    lastJ = J;

    const A = zeros(mFree, mFree);
    const g = new Float64Array(mFree);
    for (let i = 0; i < m; i++) {
      const Ji = J[i];
      for (let a = 0; a < mFree; a++) {
        const va = Ji[a];
        if (va === 0) continue;
        g[a] += va * r[i];
        for (let b = a; b < mFree; b++) A[a][b] += va * Ji[b];
      }
    }
    for (let a = 0; a < mFree; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];

    let accepted = false;
    for (let trial = 0; trial < 16; trial++) {
      const Ad = zeros(mFree, mFree);
      for (let a = 0; a < mFree; a++) {
        for (let b = 0; b < mFree; b++) Ad[a][b] = A[a][b];
        Ad[a][a] += lambda * (A[a][a] + 1e-9);
      }
      const negStep = solveSPD(Ad, g);
      if (!negStep) { lambda *= 10; continue; }
      const xNew = Float64Array.from(x);
      for (let c = 0; c < mFree; c++) xNew[freeIdx[c]] = x[freeIdx[c]] - negStep[c];
      const rNew = residualFn(xNew);
      const costNew = 0.5 * dot(rNew, rNew);
      if (Number.isFinite(costNew) && costNew < cost) {
        const rel = (cost - costNew) / Math.max(cost, 1e-300);
        const stepNorm = norm(negStep) / Math.max(norm(x) + 1e-12, 1e-12);
        x = xNew; r = rNew; cost = costNew;
        lambda = Math.max(lambda / 3, 1e-12);
        accepted = true;
        if (rel < tol || stepNorm < 1e-14) converged = true;
        break;
      }
      lambda *= 10;
      if (lambda > 1e14) break;
    }
    if (!accepted) { converged = true; break; }
    if (converged) break;
  }
  return { x, residuals: r, cost, iterations: iter, converged, jacobian: lastJ, freeIdx };
}

// Covariance of fitted physical parameters from JtJ.
function parameterCovariance(J, freeIdx, residualVariance) {
  const n = freeIdx.length;
  const A = zeros(n, n);
  for (let i = 0; i < J.length; i++) {
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) A[a][b] += J[i][a] * J[i][b];
  }
  const inv = inverse(A);
  if (!inv) return null;
  const cov = zeros(n, n);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) cov[a][b] = inv[a][b] * residualVariance;
  return cov;
}

export {
  zeros, cloneMat, transpose, matMul, matVec, dot, norm, cross, normalize3,
  solveLU, solveSPD, inverse, jacobiEigen, leastSquares, svdSmall,
  levenbergMarquardt, parameterCovariance,
};
