// Chessboard corner detection without OpenCV.
//
// 1. Structure-tensor corner response (smaller eigenvalue): chess corners are
//    saddle points with two strong orthogonal gradients, so they stand out
//    from edges and texture.
// 2. Subpixel refinement by gradient orthogonality (the cornerSubPix idea).
// 3. Lattice assembly with a bounded cols x rows homography model. Because
//    global ranking alone can miss weak corners (blur/lighting), the assembler
//    tries several seeds/orientations and refits; once a coarse grid exists we
//    re-detect each expected corner locally (predicted position -> best
//    subpixel saddle nearby), which recovers weak corners deterministically.

export function imageToGray(imageData) {
  const { data, width, height } = imageData;
  const g = new Float32Array(width * height);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) {
    g[i] = (0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2]) / 255;
  }
  return { data: g, width, height };
}

function gaussianBlur(src, w, h, sigma) {
  const radius = Math.max(1, Math.ceil(sigma * 2.5));
  const k = new Float32Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) { k[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + radius]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      if (x >= radius && x < w - radius) {
        for (let i = -radius; i <= radius; i++) s += src[row + x + i] * k[i + radius];
      } else {
        for (let i = -radius; i <= radius; i++) {
          const xx = Math.min(w - 1, Math.max(0, x + i));
          s += src[row + xx] * k[i + radius];
        }
      }
      tmp[row + x] = s;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      if (y >= radius && y < h - radius) {
        for (let i = -radius; i <= radius; i++) s += tmp[(y + i) * w + x] * k[i + radius];
      } else {
        for (let i = -radius; i <= radius; i++) {
          const yy = Math.min(h - 1, Math.max(0, y + i));
          s += tmp[yy * w + x] * k[i + radius];
        }
      }
      out[y * w + x] = s;
    }
  }
  return out;
}

function cornerResponseAndGradients(g, w, h, sigma) {
  const bl = gaussianBlur(g, w, h, sigma);
  const Ix = new Float32Array(w * h), Iy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      Ix[i] = (bl[i + 1] - bl[i - 1]) / 2;
      Iy[i] = (bl[i + w] - bl[i - w]) / 2;
    }
  }
  const Ixx = new Float32Array(w * h), Iyy = new Float32Array(w * h), Ixy = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) {
    Ixx[i] = Ix[i] * Ix[i]; Iyy[i] = Iy[i] * Iy[i]; Ixy[i] = Ix[i] * Iy[i];
  }
  const win = sigma * 2;
  const sxx = gaussianBlur(Ixx, w, h, win);
  const syy = gaussianBlur(Iyy, w, h, win);
  const sxy = gaussianBlur(Ixy, w, h, win);
  const R = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) {
    const tr = sxx[i] + syy[i];
    const det = sxx[i] * syy[i] - sxy[i] * sxy[i];
    const disc = Math.max(0, tr * tr / 4 - det);
    R[i] = tr / 2 - Math.sqrt(disc); // smaller eigenvalue
  }
  return { R, Ix, Iy, bl };
}

function cornerSubPix(Ix, Iy, w, h, x, y, winRadius = 5, iters = 10) {
  let px = x, py = y;
  for (let it = 0; it < iters; it++) {
    let a = 0, b = 0, c = 0, bx = 0, by = 0;
    const xi0 = Math.round(px), yi0 = Math.round(py);
    for (let dy = -winRadius; dy <= winRadius; dy++) {
      for (let dx = -winRadius; dx <= winRadius; dx++) {
        const x0 = xi0 + dx, y0 = yi0 + dy;
        if (x0 < 1 || y0 < 1 || x0 >= w - 1 || y0 >= h - 1) continue;
        const k = y0 * w + x0;
        const gx = Ix[k], gy = Iy[k];
        a += gx * gx; b += gx * gy; c += gy * gy;
        bx += gx * (gx * x0 + gy * y0);
        by += gy * (gx * x0 + gy * y0);
      }
    }
    const det = a * c - b * b;
    if (Math.abs(det) < 1e-12) break;
    const nx = (c * bx - b * by) / det;
    const ny = (-b * bx + a * by) / det;
    const step = Math.hypot(nx - px, ny - py);
    px = nx; py = ny;
    if (step < 1e-4) break;
  }
  return [px, py];
}

// A real chessboard internal corner has four alternating dark/light sectors
// around it, so a small circular profile through it crosses its own mean
// exactly four times and its 2nd angular harmonic dominates the 4th (opposite
// sectors share a shade). Edges give 2 crossings; blobs give 0 or many. This
// grade is what separates board corners from background clutter.
function circularProfile(g, w, h, x, y, r, N = 64) {
  const vals = new Float32Array(N);
  for (let k = 0; k < N; k++) {
    const a = (2 * Math.PI * k) / N;
    const sx = x + r * Math.cos(a), sy = y + r * Math.sin(a);
    const x0 = Math.floor(sx), y0 = Math.floor(sy);
    if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) return null;
    const fx = sx - x0, fy = sy - y0;
    vals[k] = (g[y0 * w + x0] * (1 - fx) + g[y0 * w + x0 + 1] * fx) * (1 - fy)
      + (g[(y0 + 1) * w + x0] * (1 - fx) + g[(y0 + 1) * w + x0 + 1] * fx) * fy;
  }
  return vals;
}

function checkerScore(g, w, h, x, y, radii) {
  let best = 0;
  for (const r of radii) {
    const vals = circularProfile(g, w, h, x, y, r);
    if (!vals) continue;
    const N = vals.length;
    let mean = 0;
    for (let k = 0; k < N; k++) mean += vals[k];
    mean /= N;
    let a2 = 0, b2 = 0, mn = Infinity, mx = -Infinity;
    for (let k = 0; k < N; k++) {
      const v = vals[k] - mean;
      const a = (2 * Math.PI * k) / N;
      a2 += v * Math.cos(2 * a); b2 += v * Math.sin(2 * a);
      if (vals[k] < mn) mn = vals[k];
      if (vals[k] > mx) mx = vals[k];
    }
    a2 *= 2 / N; b2 *= 2 / N;
    const A2 = Math.hypot(a2, b2);
    const contrast = mx - mn;
    let crossings = 0;
    for (let k = 0; k < N; k++) {
      const a = vals[k] - mean, b = vals[(k + 1) % N] - mean;
      if (a === 0 || b === 0) continue;
      if (Math.sign(a) !== Math.sign(b)) crossings++;
    }
    if (crossings !== 4 || contrast < 0.4 || A2 < 0.2) continue;
    // Strong internal corners reach A2 ~ 0.45-0.6; the board's outer edge and
    // other X-like clutter scrape past the gates at A2 < 0.4. Score in that
    // band so assembly can prefer strong corners while still tolerating a few
    // weak-but-real ones.
    best = Math.max(best, Math.min(1, A2 / 0.45));
  }
  return best;
}

// Public: candidate list, sorted by response. `thresholdFrac` defaults to a
// value that keeps genuine saddles; callers can lower it when needed.
export function detectCornerCandidates(gray, opts = {}) {
  const { width: w, height: h } = gray;
  const g = gray.data;
  const sigma = opts.sigma ?? 1.2;
  const { R, Ix, Iy } = cornerResponseAndGradients(g, w, h, sigma);
  const nms = opts.nmsRadius ?? 3;
  let maxR = 0;
  for (let i = 0; i < R.length; i++) if (R[i] > maxR) maxR = R[i];
  const thr = maxR * (opts.thresholdFrac ?? 0.004);
  const radii = opts.checkerRadii ?? [3, 4, 5, 6, 7, 9];
  const gradeChecker = opts.gradeChecker ?? true;
  const cands = [];
  for (let y = nms; y < h - nms; y++) {
    for (let x = nms; x < w - nms; x++) {
      const i = y * w + x;
      const v = R[i];
      if (v < thr) continue;
      let isMax = true;
      for (let dy = -nms; dy <= nms && isMax; dy++) {
        for (let dx = -nms; dx <= nms; dx++) {
          const j = (y + dy) * w + (x + dx);
          if (R[j] > v || (R[j] === v && (dy < 0 || (dy === 0 && dx < 0)))) { isMax = false; break; }
        }
      }
      if (!isMax) continue;
      const [sx, sy] = cornerSubPix(Ix, Iy, w, h, x, y, opts.subpixRadius ?? 5);
      const checker = gradeChecker ? checkerScore(g, w, h, sx, sy, radii) : 1;
      cands.push({ x: sx, y: sy, response: v, checker });
    }
  }
  cands.sort((a, b) => b.response - a.response);
  // Strongly blurred or tilted regions can give the subpixel refine several
  // nearby seeds that collapse onto the same attractor. Two entries for one
  // physical corner would let the lattice put it in two cells at once, so keep
  // only the strongest per position. Real corners are far apart (>10 px), so a
  // 1 px merge is safe.
  const uniq = [];
  for (const c of cands) {
    let dup = false;
    for (const u of uniq) {
      const dx = u.x - c.x, dy = u.y - c.y;
      if (dx * dx + dy * dy < 1) { dup = true; break; }
    }
    if (!dup) uniq.push(c);
  }
  return uniq;
}

// ---------------------------------------------------------------------------
// Lattice assembly
// ---------------------------------------------------------------------------

export function assembleGrid(candidates, cols, rows, opts = {}) {
  if (candidates.length < Math.max(6, 0.4 * cols * rows)) return null;
  // Real board corners pass the checker-profile grade strongly; background
  // saddles either fail it or scrape past with a near-zero score. When enough
  // strong corners exist, prefer them: seeding from a weak near-miss (the
  // board's outer edge, a T-junction) yields a plausible lattice anchored in
  // the wrong place.
  const strong = candidates.filter((c) => c.checker >= 0.35);
  const graded = candidates.filter((c) => c.checker > 0);
  const need = Math.max(6, 0.4 * cols * rows);
  const pool = strong.length >= need ? strong : graded.length >= need ? graded : candidates;
  const maxSeed = Math.min(pool.length, opts.maxSeeds ?? 20);
  const hash = buildHash(pool, opts.hashCell ?? 24);
  let best = null;
  for (let s = 0; s < maxSeed; s++) {
    for (const pair of seedBases(pool, s)) {
      const res = growLattice(pool, hash, s, pair.e1, pair.e2, cols, rows, opts);
      if (!res) continue;
      if (!best || better(res, best)) best = res;
      if (best.score.matched === cols * rows && best.score.err < 0.6) return best;
    }
  }
  return best;
}

export function buildHash(cands, cell) {
  const hash = new Map();
  for (let k = 0; k < cands.length; k++) {
    const key = `${Math.floor(cands[k].x / cell)},${Math.floor(cands[k].y / cell)}`;
    let arr = hash.get(key); if (!arr) { arr = []; hash.set(key, arr); }
    arr.push(k);
  }
  return { hash, cell };
}

// Nearest candidate within `tol` that is still unused. Searches the 3x3
// neighbourhood of the prediction's hash cell.
function nearestUnused(h, cands, used, px, py, tol) {
  const gx = Math.floor(px / h.cell), gy = Math.floor(py / h.cell);
  let best = -1, bestD = tol;
  for (let ox = -1; ox <= 1; ox++) {
    for (let oy = -1; oy <= 1; oy++) {
      const arr = h.hash.get(`${gx + ox},${gy + oy}`);
      if (!arr) continue;
      for (const k of arr) {
        if (used[k]) continue;
        const d = Math.hypot(cands[k].x - px, cands[k].y - py);
        if (d < bestD) { bestD = d; best = k; }
      }
    }
  }
  return best;
}

function better(a, b) {
  if (a.score.matched !== b.score.matched) return a.score.matched > b.score.matched;
  // A complete grid is a hard preference: an incomplete lattice must never win.
  const aComp = a.score.matched === a.cols * a.rows;
  const bComp = b.score.matched === b.cols * b.rows;
  if (aComp !== bComp) return aComp;
  // Among complete grids, an outer-border shift is slightly larger in extent.
  const ad = a.score.bboxDiag ?? 0, bd = b.score.bboxDiag ?? 0;
  if (Math.abs(ad - bd) > 2) return ad < bd;
  return a.score.err < b.score.err;
}

// Plausible local bases at a seed. The nearest neighbour may be a spurious
// detour (board border, clutter), so we do not anchor on it. Instead we take
// the nearest few neighbours as candidate "first axes", and for each one look
// for a near-perpendicular partner at a comparable distance. Both axis roles
// and all sign combinations are emitted so a rotated/mirrored board is covered.
export function seedBases(cands, seedIdx) {
  const p0 = cands[seedIdx];
  const neigh = [];
  for (let i = 0; i < cands.length; i++) {
    if (i === seedIdx) continue;
    const d = Math.hypot(cands[i].x - p0.x, cands[i].y - p0.y);
    if (d > 2) neigh.push({ d, c: cands[i], ang: Math.atan2(cands[i].y - p0.y, cands[i].x - p0.x) });
  }
  neigh.sort((a, b) => a.d - b.d);
  if (neigh.length < 2) return [];
  if (!(neigh[0].d > 3)) return [];

  const opts = [];
  const seenDir = new Set();
  // Try the nearest handful of candidates as the first axis.
  for (let a = 0; a < Math.min(neigh.length, 5); a++) {
    const first = neigh[a];
    const e1 = [first.c.x - p0.x, first.c.y - p0.y];
    // Partner search window scales with this first axis's distance.
    const maxRing = first.d * 1.6;
    const candidates = [];
    for (let k = 0; k < neigh.length; k++) {
      if (k === a) continue;
      const n = neigh[k];
      if (n.d < first.d * 0.6 || n.d > maxRing) continue;
      let dev = Math.abs(Math.abs(n.ang - first.ang) - Math.PI / 2);
      if (dev > Math.PI / 2) dev = Math.PI - dev;
      if (dev < 0.45) candidates.push({ dev, c: n.c, d: n.d });
    }
    candidates.sort((x, y) => x.dev - y.dev || x.d - y.d);
    for (const second of candidates.slice(0, 2)) {
      const e2 = [second.c.x - p0.x, second.c.y - p0.y];
      for (const swap of [false, true]) {
        for (const s1 of [1, -1]) {
          for (const s2 of [1, -1]) {
            const v1 = (swap ? e2 : e1).map((q) => q * s1);
            const v2 = (swap ? e1 : e2).map((q) => q * s2);
            // Deduplicate near-identical bases to keep the seed loop cheap.
            const key = [v1, v2].map((v) => `${Math.round(v[0])},${Math.round(v[1])}`).join('|');
            if (seenDir.has(key)) continue;
            seenDir.add(key);
            opts.push({ e1: v1, e2: v2 });
          }
        }
      }
    }
  }
  return opts;
}

// Grow the lattice outward from the seed cell using *local* lattice vectors
// extrapolated from already-matched neighbours. A global affine/homography
// fitted to a handful of adjacent corners does not extrapolate across a
// strong perspective; local differences do, so this reaches the far edge of
// the board where an affine seed consistently lost the last column.
//
// Cells are keyed by an unbounded integer lattice coordinate with the seed at
// (0,0). e1/e2 name the +i / +j directions; seedBases already enumerates both
// axis roles and signs, so a correct reading of the board will be produced by
// at least one seed.
export function growLattice(cands, hash, seedIdx, e1, e2, nC, nR, opts) {
  const p0 = cands[seedIdx];
  const l1 = Math.hypot(...e1), l2 = Math.hypot(...e2);
  const sep = Math.min(l1, l2);
  if (!(sep > 2)) return null;

  const matched = new Map(); // "li,lj" -> { li, lj, x, y }
  const used = new Uint8Array(cands.length);
  const put = (li, lj, k) => {
    used[k] = 1;
    const c = { li, lj, k, x: cands[k].x, y: cands[k].y };
    matched.set(`${li},${lj}`, c);
    return c;
  };

  // Find the candidate that e1/e2 point at (they were built from real
  // neighbours, so an exact-ish match exists).
  const findDir = (v) => {
    let best = -1, bd = Math.min(2, 0.2 * Math.hypot(...v));
    for (let k = 0; k < cands.length; k++) {
      if (k === seedIdx) continue;
      const d = Math.hypot(cands[k].x - p0.x - v[0], cands[k].y - p0.y - v[1]);
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  };
  const ki = findDir(e1), kj = findDir(e2);
  if (ki < 0 || kj < 0 || ki === kj) return null;
  put(0, 0, seedIdx);
  put(1, 0, ki);
  put(0, 1, kj);

  // Unit lattice directions used to pick the dominant spacing when predicting
  // off the frontier.
  const u1 = [e1[0] / l1, e1[1] / l1];
  const u2 = [e2[0] / l2, e2[1] / l2];

  // Local basis at a matched cell: forward differences when available,
  // otherwise fall back to the seed basis.
  const localBasis = (li, lj) => {
    const c0 = matched.get(`${li},${lj}`);
    let vi = null, vj = null;
    const addI = (d) => { vi = vi ? [(vi[0] + d[0]) / 2, (vi[1] + d[1]) / 2] : d; };
    const addJ = (d) => { vj = vj ? [(vj[0] + d[0]) / 2, (vj[1] + d[1]) / 2] : d; };
    let c;
    if ((c = matched.get(`${li + 1},${lj}`))) addI([c.x - c0.x, c.y - c0.y]);
    if ((c = matched.get(`${li - 1},${lj}`))) addI([c0.x - c.x, c0.y - c.y]);
    if ((c = matched.get(`${li},${lj + 1}`))) addJ([c.x - c0.x, c.y - c0.y]);
    if ((c = matched.get(`${li},${lj - 1}`))) addJ([c0.x - c.x, c0.y - c.y]);
    if (!vi) vi = e1.slice();
    if (!vj) vj = e2.slice();
    return { vi, vj };
  };

  // BFS over matched cells, extending outward.
  const queue = [...matched.values()];
  let head = 0;
  const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  while (head < queue.length && head < 5000) {
    const cell = queue[head++];
    const { li, lj } = cell;
    const { vi, vj } = localBasis(li, lj);
    for (const [di, dj] of dirs) {
      const ni = li + di, nj = lj + dj;
      const key = `${ni},${nj}`;
      if (matched.has(key)) continue;
      // Predict one step beyond this cell; when the cell behind is matched,
      // blend with the straight-line extrapolation (handles perspective).
      let px = cell.x + di * vi[0] + dj * vj[0];
      let py = cell.y + di * vi[1] + dj * vj[1];
      const behind = matched.get(`${li - di},${lj - dj}`);
      if (behind) {
        const ex = cell.x + (cell.x - behind.x);
        const ey = cell.y + (cell.y - behind.y);
        px = 0.5 * (px + ex);
        py = 0.5 * (py + ey);
      }
      // Tolerance stays below half a cell: true corners are all within a
      // fraction of a pixel of the ideal lattice, so a loose tolerance would
      // let background clutter steal cells. Perspective is handled by the
      // homography refit after growth, not by widening this window.
      const spacing = Math.hypot(di ? vi[0] : vj[0], di ? vi[1] : vj[1]);
      const tol = 0.45 * spacing;
      const k = nearestUnused(hash, cands, used, px, py, tol);
      if (k < 0) continue;
      const nc = put(ni, nj, k);
      queue.push(nc);
    }
  }

  // The seed may sit anywhere in the board. A chessboard with an outer white
  // border produces X-like saddles on its *outside* edge too, so the grown
  // lattice is typically (nC+2) x (nR+2). Choose the nC x nR window that the
  // lattice actually supports, using OpenCV's criterion: an internal corner
  // set has neighbours on ALL FOUR sides. A window shifted one cell out has a
  // free outside edge, so it loses. Count window cells, plus a bonus for
  // neighbours lying immediately outside each window edge (the boundary ring).
  let liMin = Infinity, ljMin = Infinity, liMax = -Infinity, ljMax = -Infinity;
  for (const c of matched.values()) {
    liMin = Math.min(liMin, c.li); liMax = Math.max(liMax, c.li);
    ljMin = Math.min(ljMin, c.lj); ljMax = Math.max(ljMax, c.lj);
  }
  const has = (li, lj) => matched.has(`${li},${lj}`);
  let bestOff = null, bestWeight = -1;
  for (let oi = liMin - nC + 1; oi <= liMax; oi++) {
    for (let oj = ljMin - nR + 1; oj <= ljMax; oj++) {
      let weight = 0, count = 0;
      for (let gi = 0; gi < nC; gi++) {
        for (let gj = 0; gj < nR; gj++) {
          const c = matched.get(`${gi + oi},${gj + oj}`);
          if (!c) continue;
          count++;
          weight += 1 + (cands[c.k]?.checker ?? 0);
        }
      }
      // Ring support: a genuine board window is bounded by real corners on
      // every side; an outward-shifted window has a bare edge.
      let ring = 0;
      for (let gi = 0; gi < nC; gi++) {
        if (has(gi + oi, -1 + oj)) ring++;
        if (has(gi + oi, nR + oj)) ring++;
      }
      for (let gj = 0; gj < nR; gj++) {
        if (has(-1 + oi, gj + oj)) ring++;
        if (has(nC + oi, gj + oj)) ring++;
      }
      const score = weight + 0.5 * ring;
      if (count > 0 && score > bestWeight) { bestWeight = score; bestOff = { oi, oj }; }
    }
  }
  if (!bestOff) return null;

  // Refine the window by fitting a homography on the cells it contains and
  // re-matching every predicted cell against the full candidate list. This
  // fixes a near-miss window (a row/column off) and recovers cells whose
  // checker grade was zero — e.g. a corner whose profile straddles the board
  // edge and fails the circular test even though the saddle is exact.
  let win = bestOff;
  let bestGrid = collectWindow(matched, cands, win, nC, nR);
  let bestErr = bestGrid.length >= 4 ? latticeRms(bestGrid) : Infinity;
  for (let pass = 0; pass < 3; pass++) {
    if (bestGrid.length < 4) break;
    const H = fitHomography(bestGrid.map((m) => [m.i, m.j]), bestGrid.map((m) => [m.x, m.y]));
    if (!H) break;
    const refined = rematchWindow(cands, H, nC, nR, sep);
    const err = refined.length >= 4 ? latticeRms(refined) : Infinity;
    const betterCount = refined.length > bestGrid.length;
    const sameCount = refined.length === bestGrid.length;
    if (betterCount || (sameCount && err < bestErr)) {
      bestGrid = refined;
      bestErr = err;
    } else break;
  }
  // If refinement produced a full grid, it supersedes the window selection.
  let grid;
  if (bestGrid.length >= 4) {
    grid = bestGrid.map((m) => ({ i: m.i, j: m.j, x: m.x, y: m.y }));
  } else {
    grid = collectWindow(matched, cands, win, nC, nR).map((m) => ({ i: m.i, j: m.j, x: m.x, y: m.y }));
  }
  if (grid.length < 4) return null;

  // Canonicalise the index labels with a HANDEDNESS-PRESERVING rule.
  //
  // The lattice growth may return labels that are any element of the
  // rectangle's symmetry group relative to the physical board. For one image
  // that is harmless (the homography absorbs it), but when several views are
  // calibrated together they must all use the same object frame. Projection of
  // a front-facing plane preserves orientation, so the true projected +i/+j
  // basis has a fixed cross-product sign. Enforcing that sign here picks the
  // physical parity in every view; the remaining identity-or-180° freedom is a
  // genuine rotation of the board and is absorbed by the pose.
  //
  // Never flip exactly one axis for cosmetic reasons — that is a reflection and
  // would disagree with other views.
  const axes = latticeAxes(grid);
  if (axes.ei && axes.ej) {
    let cross = axes.ei[0] * axes.ej[1] - axes.ei[1] * axes.ej[0];
    if (cross < 0) {
      // Reverse one axis to restore the physical handedness. (This is the only
      // place a reflection is allowed, and it is forced by geometry, not
      // cosmetic.)
      for (const g of grid) g.i = nC - 1 - g.i;
      axes.ei = [-axes.ei[0], -axes.ei[1]];
    }
    // Among the remaining parity-preserving labelings only a 180° rotation is
    // left (flipping BOTH axes). Apply at most one, so +i ends up pointing
    // right (or, for a near-vertical +i axis, +j downward).
    const flipBoth = axes.ei[0] < 0
      || (Math.abs(axes.ei[0]) < 1e-9 && axes.ej[1] < 0);
    if (flipBoth) {
      for (const g of grid) { g.i = nC - 1 - g.i; g.j = nR - 1 - g.j; }
    }
  }

  const complete = grid.length === nC * nR;
  const err = latticeRms(grid);
  // Bounding-box diagonal in image space. A window shifted one cell outward
  // onto the board's outer border has the same cell count but a larger
  // extent; prefer the smaller one when otherwise tied.
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const g of grid) {
    minX = Math.min(minX, g.x); minY = Math.min(minY, g.y);
    maxX = Math.max(maxX, g.x); maxY = Math.max(maxY, g.y);
  }
  const bboxDiag = Math.hypot(maxX - minX, maxY - minY);
  grid.sort((p, q) => p.j - q.j || p.i - q.i);
  return {
    points: grid.map((m) => [m.x, m.y]),
    gridIJ: grid.map((m) => [m.i, m.j]),
    cols: nC, rows: nR,
    homography: grid.length >= 4 ? fitHomography(grid.map((m) => [m.i, m.j]), grid.map((m) => [m.x, m.y])) : null,
    partial: !complete,
    score: { matched: grid.length, err: complete ? err : err + 1e6, bboxDiag },
  };
}

// Median screen-space step for +i and +j across the grid. Median (not mean)
// keeps isolated mis-detections from rotating the estimate.
function latticeAxes(grid) {
  const at = new Map(grid.map((g) => [`${g.i},${g.j}`, g]));
  const di = [], dj = [];
  for (const g of grid) {
    const r = at.get(`${g.i + 1},${g.j}`);
    if (r) di.push([r.x - g.x, r.y - g.y]);
    const d = at.get(`${g.i},${g.j + 1}`);
    if (d) dj.push([d.x - g.x, d.y - g.y]);
  }
  const med = (arr) => {
    if (!arr.length) return null;
    const xs = arr.map((p) => p[0]).sort((a, b) => a - b);
    const ys = arr.map((p) => p[1]).sort((a, b) => a - b);
    return [xs[xs.length >> 1], ys[ys.length >> 1]];
  };
  return { ei: med(di), ej: med(dj) };
}

function collectWindow(matched, cands, off, nC, nR) {
  const out = [];
  for (let i = 0; i < nC; i++) {
    for (let j = 0; j < nR; j++) {
      const c = matched.get(`${i + off.oi},${j + off.oj}`);
      if (c) out.push({ i, j, x: c.x, y: c.y });
    }
  }
  return out;
}

// For every grid cell, predict with the homography and take the nearest unused
// candidate within a tight window, regardless of checker grade.
function rematchWindow(cands, H, nC, nR, sep) {
  const used = new Uint8Array(cands.length);
  const out = [];
  const tol = sep * 0.5;
  for (let j = 0; j < nR; j++) {
    for (let i = 0; i < nC; i++) {
      const pr = applyH(H, i, j);
      let best = -1, bestD = tol;
      for (let k = 0; k < cands.length; k++) {
        if (used[k]) continue;
        const d = Math.hypot(cands[k].x - pr.x, cands[k].y - pr.y);
        if (d < bestD) { bestD = d; best = k; }
      }
      if (best >= 0) { used[best] = 1; out.push({ i, j, x: cands[best].x, y: cands[best].y }); }
    }
  }
  return out;
}

function latticeRms(pts) {
  if (pts.length < 4) return Infinity;
  const H = fitHomography(pts.map((m) => [m.i, m.j]), pts.map((m) => [m.x, m.y]));
  if (!H) return Infinity;
  let sq = 0;
  for (const m of pts) {
    const pr = applyH(H, m.i, m.j);
    sq += (pr.x - m.x) ** 2 + (pr.y - m.y) ** 2;
  }
  return Math.sqrt(sq / pts.length);
}

export function fitHomography(src, dst) {
  let cx = 0, cy = 0, cu = 0, cv = 0;
  const n = src.length;
  for (let i = 0; i < n; i++) { cx += src[i][0]; cy += src[i][1]; cu += dst[i][0]; cv += dst[i][1]; }
  cx /= n; cy /= n; cu /= n; cv /= n;
  const A = [];
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - cx, y = src[i][1] - cy;
    const u = dst[i][0] - cu, v = dst[i][1] - cv;
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, -u]);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, -v]);
  }
  const AtA = zeros(8, 8), Atb = new Float64Array(8);
  for (const row of A) {
    for (let a = 0; a < 8; a++) {
      Atb[a] -= row[a] * row[8];
      for (let b = 0; b < 8; b++) AtA[a][b] += row[a] * row[b];
    }
  }
  const h = solve(AtA, Atb);
  if (!h) return null;
  // Undo the centring exactly by composing H = D * Hc * S with
  //   S: x' = x - c,  D: u = u' + c.
  // The DLT solved for u' = (h0 x' + h1 y' + h2)/w with w = h6 x' + h7 y' + 1.
  // Expanding gives h8 = 1 - h6 cx - h7 cy and, crucially, cu*h6 / cu*h7 in
  // the numerators (not just cu*h8); dropping those only looks correct for
  // near-affine boards and corrupts prediction on tilted ones.
  const h8 = 1 - (h[6] * cx + h[7] * cy);
  return [
    [h[0] + cu * h[6], h[1] + cu * h[7], h[2] - (h[0] * cx + h[1] * cy) + cu * h8],
    [h[3] + cv * h[6], h[4] + cv * h[7], h[5] - (h[3] * cx + h[4] * cy) + cv * h8],
    [h[6], h[7], h8],
  ];
}

function applyH(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return { x: (H[0][0] * x + H[0][1] * y + H[0][2]) / w, y: (H[1][0] * x + H[1][1] * y + H[1][2]) / w };
}

function zeros(n, m) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = new Float64Array(m); return a; }
function solve(A, b) {
  const n = A.length;
  const M = A.map((r) => Float64Array.from(r));
  const x = Float64Array.from(b);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;
    if (piv !== c) { const t = M[piv]; M[piv] = M[c]; M[c] = t; const tb = x[piv]; x[piv] = x[c]; x[c] = tb; }
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      if (f === 0) continue;
      for (let k = c; k < n; k++) M[r][k] -= f * M[c][k];
      x[r] -= f * x[c];
    }
  }
  const out = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= M[i][k] * out[k];
    out[i] = s / M[i][i];
  }
  return out;
}

export function findChessboard(gray, cols, rows, opts = {}) {
  // Keep a generous candidate list; the assembler is the part that must be
  // robust, and true corners can rank low on blurred/dark images.
  const cands = detectCornerCandidates(gray, opts);
  const grid = assembleGrid(cands, cols, rows, opts);
  return { grid, candidates: cands };
}
