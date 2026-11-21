// Synthetic + real datasets for the workbench.
//
// The synthetic dataset is a zoom + crop sequence whose lens is known exactly,
// so every fitted quantity can be scored against ground truth. The real dataset
// is a set of chessboard photographs (OpenCV's sample images, Apache-2.0) with
// no truth camera; it is used to check the detector and the map on real optics.

import { buildZoomCropSequence, renderCheckerboardImage } from '../core/synth.js';
import { findChessboard } from '../core/detect.js';
import { imageToGray } from '../core/detect.js';

const PITCH = 36 / 640;
const SQUARE = 20; // mm; arbitrary but consistent object-space scale
const BOARD = { cols: 9, rows: 6, squareSize: SQUARE };
const BOARD_W = 9 * SQUARE;

// OpenCV's sample boards are 10x7 squares = 9x6 internal corners. Detection
// must be asked for that many, otherwise the assembler happily fits a truncated
// sub-lattice and the last row/column never enters the calibration.
const REAL_BOARD = { cols: 10, rows: 7 };
const REAL_CORNERS = { cols: REAL_BOARD.cols - 1, rows: REAL_BOARD.rows - 1 };

// Board at distance z fills `frac` of an image `imgWidth` px wide.
function zFor(f_mm, imgWidth, frac) {
  return (BOARD_W * f_mm) / (frac * imgWidth * PITCH);
}

function poseAt(f_mm, z, rx, ry, rz) {
  return { rx, ry, rz, t: [0, 0, z] };
}

// The truth lens is dominated by a fourth-order radial term. That matters for
// the demo: the four small, near-central wide boards have essentially no
// leverage on k2, so the single strongly tilted view is the only observation
// that constrains it. On the per-focal path removing that view lets k2 run
// away; the shared path keeps it anchored through the other segments.
export const SYNTH_TRUTH = {
  principalMM: { px: 0.80, py: -0.50 },
  distortion: { k1: 0, k2: 2.0e-7, k3: 0, p1: 1.0e-4, p2: -0.8e-4 },
  sensorMM: 36,
  pitch: PITCH,
};

// Scene: a wide full-frame readout and two cropped, zoomed readouts. The mid
// and tele boards stay in the centre of their frames, which is what makes the
// per-focal fit extrapolate garbage at the periphery while the shared physical
// sensor stays anchored to the wide segment.
//
// The wide segment deliberately has four small, only mildly tilted boards that
// sit near the image centre, plus one large, strongly tilted board that is the
// ONLY observation reaching the sensor corners. That is what makes the
// interactive demo honest: delete the tilted view and the wide fit loses its
// only peripheral constraint. On the per-focal path the radial term then
// overshoots, the actual map folds, and the safe crop must shrink. On the
// shared path the other segments keep the shared distortion anchored, so the
// crop barely moves.
export function buildSyntheticScene() {
  const W = 640, H = 480;
  const zWide = zFor(30, W, 0.19);
  return {
    width: W, height: H, sensorMM: SYNTH_TRUTH.sensorMM,
    principalMM: SYNTH_TRUTH.principalMM, distortion: SYNTH_TRUTH.distortion,
    board: BOARD, seed: 2024,
    segments: [
      {
        focalId: 'wide', f_mm: 30, width: W, height: H, cornerNoisePx: 0.2,
        poses: [
          poseAt(30, zWide * 0.92, 6, -3, 5),
          poseAt(30, zWide * 1.12, -5, 4, -8),
          poseAt(30, zWide * 1.28, 2, 6, 13),
          poseAt(30, zWide * 1.00, -3, -5, -14),
          // near, strongly tilted: the only view that reaches the sensor corners
          poseAt(30, zFor(30, W, 0.95), 40, -36, 10),
        ],
      },
      {
        focalId: 'mid', f_mm: 50, width: 480, height: 360,
        crop: { originX_mm: 2.5, originY_mm: -1.5 }, cornerNoisePx: 0.2,
        poses: [
          poseAt(50, zFor(50, 480, 0.44) * 0.95, 12, -14, 4),
          poseAt(50, zFor(50, 480, 0.44) * 1.12, -16, 12, -8),
          poseAt(50, zFor(50, 480, 0.44) * 1.30, 18, 9, 13),
          poseAt(50, zFor(50, 480, 0.44) * 1.02, -11, -18, -14),
        ],
      },
      {
        focalId: 'tele', f_mm: 85, width: 320, height: 240,
        crop: { originX_mm: -3.0, originY_mm: 2.2 }, cornerNoisePx: 0.2,
        poses: [
          poseAt(85, zFor(85, 320, 0.44) * 0.95, 11, -13, 5),
          poseAt(85, zFor(85, 320, 0.44) * 1.12, -17, 11, -9),
          poseAt(85, zFor(85, 320, 0.44) * 1.30, 19, 10, 14),
          poseAt(85, zFor(85, 320, 0.44) * 1.02, -12, -19, -12),
        ],
      },
    ],
  };
}

export function previewView(seg, view, board) {
  // Render one view of a synthetic segment to a Float32 grayscale raster.
  return renderCheckerboardImage(seg.camera, view.R, view.t, board);
}

// Build the synthetic dataset: render every view, detect the board, and keep
// both the ground-truth correspondence and the detected one so the UI can
// choose (the task asks for a genuinely-detected pipeline, so the detector
// result is what gets calibrated by default).
export function loadSynthetic() {
  const config = buildSyntheticScene();
  const seq = buildZoomCropSequence(config);
  const segments = [];
  let vi = 0;
  for (const seg of seq.segments) {
    const views = [];
    seg.views.forEach((v, k) => {
      const grayData = renderCheckerboardImage(seg.camera, v.R, v.t, config.board);
      const gray = { data: grayData, width: seg.camera.width, height: seg.camera.height };
      const { grid } = findChessboard(gray, BOARD.cols - 1, BOARD.rows - 1, {});
      views.push({
        id: `${seg.focalId}#${k}`,
        index: vi++,
        segmentId: seg.focalId,
        gray,
        grid,
        truthPoints: v.allImagePoints,
        truthVisible: v.visible,
        width: seg.camera.width,
        height: seg.camera.height,
        deleted: false,
        excluded: new Set(),
      });
    });
    segments.push({
      focalId: seg.focalId,
      f_mm: seg.f_mm,
      camera: seg.camera,
      crop: seg.crop,
      originX_mm: seg.crop?.originX_mm ?? 0,
      originY_mm: seg.crop?.originY_mm ?? 0,
      width: seg.camera.width,
      height: seg.camera.height,
      views,
    });
  }
  return {
    kind: 'synthetic',
    segments,
    truth: {
      camera: null,
      principalMM: SYNTH_TRUTH.principalMM,
      distortion: SYNTH_TRUTH.distortion,
      sensorMM: SYNTH_TRUTH.sensorMM,
      pitchX: PITCH,
      pitchY: PITCH,
      board: BOARD,
      config,
    },
  };
}

// ---------------------------------------------------------------------------
// Real photographs
// ---------------------------------------------------------------------------

const REAL_NAMES = ['left01', 'left02', 'left03', 'left04', 'left05', 'left06',
  'left07', 'left08', 'left09', 'left11', 'left12', 'left13', 'left14'];

async function loadImageData(url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const id = ctx.getImageData(0, 0, c.width, c.height);
  return { data: id.data, width: id.width, height: id.height };
}

export async function loadReal(baseUrl = '../../data/samples/') {
  const first = await loadImageData(`${baseUrl}${REAL_NAMES[0]}.jpg`);
  const width = first.width, height = first.height;
  const views = [];
  for (let k = 0; k < REAL_NAMES.length; k++) {
    const name = REAL_NAMES[k];
    const imgData = k === 0 ? first : await loadImageData(`${baseUrl}${name}.jpg`);
    const gray = imageToGray(imgData);
    const { grid } = findChessboard(gray, REAL_CORNERS.cols, REAL_CORNERS.rows, {});
    views.push({
      id: name,
      index: k,
      segmentId: 'cam',
      gray,
      grid,
      truthPoints: null,
      truthVisible: null,
      width,
      height,
      deleted: false,
      excluded: new Set(),
    });
  }
  const sensorMM = 36;
  const segments = [{
    focalId: 'cam', f_mm: null,
    originX_mm: 0, originY_mm: 0,
    width, height,
    views,
  }];
  return {
    kind: 'real',
    segments,
    truth: {
      camera: null,
      principalMM: null,
      distortion: null,
      sensorMM,
      pitchX: sensorMM / width,
      pitchY: sensorMM / width,
      board: REAL_BOARD,
      attribution: 'OpenCV sample images, Apache-2.0 (data/samples/LICENSE-OpenCV-Apache-2.0.txt)',
    },
  };
}

// Split real views into fit and held-out sets. The held-out views are never
// passed to the solver, so their reprojection error and board-line straightness
// are honest generalisation checks. A deterministic interleave avoids biasing
// the split to any particular board orientation.
export function splitHeldOut(views, holdOutCount = 3) {
  const usable = views.filter((v) => v.grid && !v.deleted);
  const ordered = usable.slice().sort((a, b) => a.index - b.index);
  const held = new Set();
  const step = ordered.length / holdOutCount;
  for (let i = 0; i < holdOutCount; i++) held.add(Math.min(ordered.length - 1, Math.floor(i * step + step / 2)));
  const heldOut = [];
  const fit = [];
  ordered.forEach((v, i) => (held.has(i) ? heldOut : fit).push(v));
  return { fit, heldOut };
}
