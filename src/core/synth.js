// Synthetic ground-truth scene generator.
//
// We render a checkerboard (and a held-out set of straight lines that are
// NEVER passed to the solver) with an exactly known physical lens. The
// renderer works on the sensor plane in mm using the same lens model the
// calibrator uses, so any disagreement between truth and fit is a solver
// error, not a rendering convention mismatch.

import {
  distortNormalized, projectPoint, intrinsicsToSensor, normalizedToPixel,
} from './lens.js';

// Build object grid: (cols-1) x (rows-1) internal corners on a board that is
// `cols` x `rows` SQUARES. Internal corner lines sit at square boundaries, so
// the first corner is one square in from the board corner (at squareSize, not
// 0). This matches what renderCheckerboardImage draws — otherwise every truth
// point would land on the board's outer edge instead of an X-corner.
export function checkerboardObjectPoints(cols, rows, squareSize = 1) {
  const pts = [];
  for (let y = 1; y < rows; y++) {
    for (let x = 1; x < cols; x++) pts.push([x * squareSize, y * squareSize, 0]);
  }
  return pts;
}

// Rotation from Euler angles (degrees).
export function eulerRotation(rxDeg, ryDeg, rzDeg) {
  const rx = (rxDeg * Math.PI) / 180, ry = (ryDeg * Math.PI) / 180, rz = (rzDeg * Math.PI) / 180;
  const cx = Math.cos(rx), sx = Math.sin(rx);
  const cy = Math.cos(ry), sy = Math.sin(ry);
  const cz = Math.cos(rz), sz = Math.sin(rz);
  const Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]];
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
  return mul3(mul3(Rz, Ry), Rx);
}
function mul3(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}

// A "camera" is a full physical description: sensor + focal length + crop.
// `sensorPixels` is the full sensor readout width in pixels; pitch is derived
// from it and stays constant across crops. A cropped image has fewer pixels
// (width/height) but the same pitch, with `originX_mm/originY_mm` locating the
// crop centre on the sensor.
export function makeCamera({ width, height, sensorMM = 36, sensorPixels, fx_mm, fy_mm,
  px_mm, py_mm, originX_mm = 0, originY_mm = 0, distortion = {}, sensorHeightMM }) {
  const fullW = sensorPixels ?? width;
  const pitchX = sensorMM / fullW;
  const fullH = sensorPixels ? Math.round((sensorPixels * height) / width) : height;
  const heightMM = sensorHeightMM ?? (sensorMM * fullH) / fullW;
  const pitchY = heightMM / fullH;
  return {
    width, height, pitchX, pitchY,
    fx_mm, fy_mm: fy_mm ?? fx_mm,
    px_mm, py_mm,
    originX_mm, originY_mm,
    widthMM: width * pitchX, heightMM: height * pitchY,
    fullWidthMM: sensorMM, fullHeightMM: heightMM,
    distortion: {
      k1: distortion.k1 ?? 0, k2: distortion.k2 ?? 0, k3: distortion.k3 ?? 0,
      p1: distortion.p1 ?? 0, p2: distortion.p2 ?? 0,
    },
  };
}

// Project a 3D object point (mm in board coords) through a pose into pixels.
export function projectObjectPoint(X, Y, Z, R, t, cam) {
  const Xc = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0];
  const Yc = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1];
  const Zc = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2];
  return projectPoint(Xc, Yc, Zc, cam, cam.distortion);
}

// Render a view of the board. Returns { imagePoints, objectPoints, visible,
// pose } with imagePoints in the *camera's own pixel frame* (accounting for the
// crop origin). Corners are only marked visible if their projection is inside
// the image and the board faces the camera.
export function renderView(cam, board, pose) {
  const { R, t } = pose;
  const imagePoints = [];
  const visible = [];
  for (const P of board.objectPoints) {
    const pr = projectObjectPoint(P[0], P[1], P[2], R, t, cam);
    imagePoints.push([pr.u, pr.v]);
    visible.push(
      pr.u >= 2 && pr.v >= 2 && pr.u <= cam.width - 3 && pr.v <= cam.height - 3 && pr.distortionDet > 0,
    );
  }
  return { imagePoints, objectPoints: board.objectPoints, visible, pose, width: cam.width, height: cam.height };
}

// ---------------------------------------------------------------------------
// Held-out straight lines (never used for fitting)
// ---------------------------------------------------------------------------
//
// A physical line in the world is projected through the true lens. We sample
// the *ideal* pinhole image of that line, then push each sample through the
// inverse-of-truth to get where it actually appears. After undistortion with
// a candidate model, those samples should be collinear; the maximum deviation
// from the best-fit line is the residual bending.

export function heldOutLines(cam, R, t, lines, samplesPerLine = 40) {
  const out = [];
  for (const line of lines) {
    // Line defined by two 3D endpoints; interpolate.
    const [a, b] = line;
    const pts = [];
    const ideal = [];
    for (let i = 0; i < samplesPerLine; i++) {
      const s = i / (samplesPerLine - 1);
      const X = a[0] + s * (b[0] - a[0]);
      const Y = a[1] + s * (b[1] - a[1]);
      const Z = a[2] + s * (b[2] - a[2]);
      const Xc = R[0][0] * X + R[0][1] * Y + R[0][2] * Z + t[0];
      const Yc = R[1][0] * X + R[1][1] * Y + R[1][2] * Z + t[1];
      const Zc = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + t[2];
      if (Zc <= 1e-9) continue;
      const pr = projectPoint(Xc, Yc, Zc, cam, cam.distortion);
      // Ideal pinhole projection (no distortion) for reference.
      const pitchX = cam.pitchX, pitchY = cam.pitchY;
      const idealU = (cam.originX_mm + (cam.fx_mm * Xc) / Zc - cam.px_mm) / pitchX + (cam.width - 1) / 2;
      const idealV = (cam.originY_mm + (cam.fy_mm * Yc) / Zc - cam.py_mm) / pitchY + (cam.height - 1) / 2;
      pts.push([pr.u, pr.v]);
      ideal.push([idealU, idealV]);
    }
    out.push({ observed: pts, ideal, endpoints: line });
  }
  return out;
}

// Fit a line to 2D points and return max/mean perpendicular deviation.
export function lineDeviation(points) {
  const n = points.length;
  let mx = 0, my = 0;
  for (const p of points) { mx += p[0]; my += p[1]; }
  mx /= n; my /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of points) {
    const dx = p[0] - mx, dy = p[1] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  // Principal direction of the smallest eigenvalue = normal.
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const nx = Math.cos(theta + Math.PI / 2);
  const ny = Math.sin(theta + Math.PI / 2);
  let maxDev = 0, sumAbs = 0, sumSq = 0;
  for (const p of points) {
    const d = Math.abs((p[0] - mx) * nx + (p[1] - my) * ny);
    maxDev = Math.max(maxDev, d);
    sumAbs += d;
    sumSq += d * d;
  }
  return { maxDeviationPx: maxDev, meanDeviationPx: sumAbs / n, rmsDeviationPx: Math.sqrt(sumSq / n) };
}

// ---------------------------------------------------------------------------
// Scene builders
// ---------------------------------------------------------------------------

// A zoom + crop sequence. Each segment gets a list of poses. `crops` describes
// the crop window in *full sensor* coordinates. Cameras share one physical
// lens (fx_mm, px_mm, py_mm, distortion) but each segment may zoom and crop.
//
// Returns { segments, truth } where each segment carries its camera and views
// and truth records the exact parameters for later error checks.
export function buildZoomCropSequence(config) {
  const {
    width, height, sensorMM = 36, principalMM = { px: 0, py: 0 },
    distortion,
    segments: segConfigs,
    board = { cols: 9, rows: 6, squareSize: 1 },
    seed = 1,
  } = config;
  const rng = mulberry32(seed);
  const objPts = checkerboardObjectPoints(board.cols, board.rows, board.squareSize);
  const objectPoints = objPts;
  const segments = [];
  for (const sc of segConfigs) {
    const cam = makeCamera({
      width: sc.width ?? width, height: sc.height ?? height,
      sensorMM, sensorPixels: width,
      fx_mm: sc.f_mm, fy_mm: sc.f_mm,
      px_mm: principalMM.px, py_mm: principalMM.py,
      originX_mm: sc.crop ? sc.crop.originX_mm : 0,
      originY_mm: sc.crop ? sc.crop.originY_mm : 0,
      distortion,
    });
    const views = [];
    for (const ps of sc.poses) {
      const R = eulerRotation(ps.rx ?? 0, ps.ry ?? 0, ps.rz ?? 0);
      const t = ps.t ?? [0, 0, ps.z ?? 600];
      const jitter = sc.cornerNoisePx ?? 0;
      const raw = renderView(cam, { objectPoints }, { R, t });
      const noisy = raw.imagePoints.map((p) => [
        p[0] + (rng() - 0.5) * 2 * jitter,
        p[1] + (rng() - 0.5) * 2 * jitter,
      ]);
      // Keep only corners that land inside this (possibly cropped) image.
      const keptObj = [];
      const keptImg = [];
      for (let i = 0; i < raw.imagePoints.length; i++) {
        if (raw.visible[i]) { keptObj.push(objectPoints[i]); keptImg.push(noisy[i]); }
      }
      views.push({
        objectPoints: keptObj, imagePoints: keptImg,
        width: cam.width, height: cam.height,
        allImagePoints: raw.imagePoints, visible: raw.visible,
        R, t,
      });
    }
    segments.push({
      focalId: sc.focalId, f_mm: sc.f_mm,
      crop: sc.crop ?? { originX_mm: 0, originY_mm: 0 },
      camera: cam, views,
    });
  }
  return {
    segments,
    truth: { width, height, sensorMM, principalMM, distortion, board, camera: { width, height, sensorMM } },
  };
}

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Render the synthetic scene to a Float32 grayscale raster so the UI can
// display a "photo" and so the detector can be run on it. A hard parity test
// per pixel would alias badly along the square edges and bias subpixel
// corners, so each pixel is averaged over a supersample grid.
export function renderCheckerboardImage(cam, R, t, board, out = null, opts = {}) {
  const { width, height } = cam;
  const ss = opts.supersample ?? 3;
  const img = out ?? new Float32Array(width * height);
  const inv = invertRigid(R, t);
  const sample = (fu, fv) => {
    // Ideal pixel -> normalised -> undistort -> world ray -> board plane z=0.
    // Distortion coefficients are in sensor-mm, so scale by focal length
    // exactly as projectPoint/normalizedToPixel do; otherwise the rendered
    // image and the truth projections use different models.
    const xn = ((cam.originX_mm + (fu - (width - 1) / 2) * cam.pitchX) - cam.px_mm) / cam.fx_mm;
    const yn = ((cam.originY_mm + (fv - (height - 1) / 2) * cam.pitchY) - cam.py_mm) / cam.fy_mm;
    const und = undistortNormalizedLocal(cam.fx_mm * xn, cam.fy_mm * yn, cam.distortion);
    const rx = und.x / cam.fx_mm;
    const ry = und.y / cam.fy_mm;
    // Camera centre C = -R^T t (this is inv.t). Ray direction in world is
    // R^T (rx, ry, 1). Intersect that ray with the board plane z=0.
    const dx = inv.R[0][0] * rx + inv.R[0][1] * ry + inv.R[0][2];
    const dy = inv.R[1][0] * rx + inv.R[1][1] * ry + inv.R[1][2];
    const dz = inv.R[2][0] * rx + inv.R[2][1] * ry + inv.R[2][2];
    if (Math.abs(dz) > 1e-9) {
      const s = -inv.t[2] / dz;
      if (s > 0) {
        const wx = inv.t[0] + s * dx;
        const wy = inv.t[1] + s * dy;
        if (wx >= 0 && wy >= 0 && wx < board.cols * board.squareSize && wy < board.rows * board.squareSize) {
          const sx = Math.floor(wx / board.squareSize);
          const sy = Math.floor(wy / board.squareSize);
          return (sx + sy) % 2 === 0 ? 0.05 : 0.95;
        }
      }
    }
    return dz > 1e-9 ? 0.35 : 0.2; // background behind / around the board
  };
  const invSS = 1 / ss;
  for (let v = 0; v < height; v++) {
    for (let u = 0; u < width; u++) {
      let acc = 0;
      for (let j = 0; j < ss; j++) {
        for (let i = 0; i < ss; i++) acc += sample(u + (i + 0.5) * invSS - 0.5, v + (j + 0.5) * invSS - 0.5);
      }
      img[v * width + u] = acc / (ss * ss);
    }
  }
  return img;
}

function invertRigid(R, t) {
  const Rt = [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]];
  const tw = [
    -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
    -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
    -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
  ];
  return { R: Rt, t: tw };
}

function undistortNormalizedLocal(xd, yd, k) {
  let x = xd, y = yd;
  const f = (x, y) => {
    const r2 = x * x + y * y;
    const radial = 1 + k.k1 * r2 + k.k2 * r2 * r2 + k.k3 * r2 * r2 * r2;
    return {
      xd: x * radial + 2 * k.p1 * x * y + k.p2 * (r2 + 2 * x * x),
      yd: y * radial + k.p1 * (r2 + 2 * y * y) + 2 * k.p2 * x * y,
    };
  };
  for (let it = 0; it < 30; it++) {
    const d = f(x, y);
    const ex = d.xd - xd, ey = d.yd - yd;
    if (ex * ex + ey * ey < 1e-16) break;
    const r2 = x * x + y * y;
    const radial = 1 + k.k1 * r2 + k.k2 * r2 * r2 + k.k3 * r2 * r2 * r2;
    const dr = k.k1 + 2 * k.k2 * r2 + 3 * k.k3 * r2 * r2;
    const j00 = radial + 2 * x * x * dr + 2 * k.p1 * y + 6 * k.p2 * x;
    const j01 = 2 * x * y * dr + 2 * k.p1 * x + 2 * k.p2 * y;
    const j10 = 2 * x * y * dr + 2 * k.p1 * x + 2 * k.p2 * y;
    const j11 = radial + 2 * y * y * dr + 6 * k.p1 * y + 2 * k.p2 * x;
    const det = j00 * j11 - j01 * j10;
    if (!(Math.abs(det) > 1e-14)) break;
    x -= (j11 * ex - j01 * ey) / det;
    y -= (-j10 * ex + j00 * ey) / det;
  }
  return { x, y };
}
