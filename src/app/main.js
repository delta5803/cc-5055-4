// Workbench UI.
//
// Renders the selected view with overlays (detected corners, held-out lines,
// distortion field, the safe-crop window derived from the actual map), lets the
// user exclude corners and delete whole views, and exports a directly usable
// destination->source map plus a report of the verification numbers.

import { loadSynthetic, loadReal, splitHeldOut } from './dataset.js';
import { solve, resampleView, toRGBA } from './solve.js';
import { defaultIdealLines, truthObservedPixel } from '../core/evaluate.js';
import {
  undistortPixel, distortionFieldAt, normalizedToPixel,
  sensorXFromPixel, sensorYFromPixel,
} from '../core/lens.js';
import { encodePng, mapToJSON, sourceAt } from '../core/resample.js';

const $ = (id) => document.getElementById(id);

const state = {
  dataset: null,
  path: 'shared',
  model: 'poly5',
  windowMode: 'auto',
  result: null,
  viewId: null,
  mode: 'raw',
  showCorners: true,
  showLines: true,
  showField: false,
  showWindow: true,
  showMapValid: false,
  heldOut: new Set(),
  busy: false,
  prevWindows: new Map(),   // focalId -> window from the previous solve
  resampleCache: new Map(),
  solveId: 0,
};

const canvas = $('canvas');
const ctx = canvas.getContext('2d');

// ---------------------------------------------------------------------------
// Lookup helpers
// ---------------------------------------------------------------------------

function currentView() {
  if (!state.dataset) return null;
  for (let si = 0; si < state.dataset.segments.length; si++) {
    const seg = state.dataset.segments[si];
    const view = seg.views.find((x) => x.id === state.viewId);
    if (view) return { seg, view, si };
  }
  return null;
}

function currentResult() {
  const cur = currentView();
  if (!cur || !state.result?.perSegment) return null;
  return state.result.perSegment[cur.si] ?? null;
}

function setStatus(text, cls = '') {
  const el = $('status');
  el.textContent = text;
  el.className = 'status ' + cls;
}

// ---------------------------------------------------------------------------
// Canvas rendering
// ---------------------------------------------------------------------------

function draw() {
  const cur = currentView();
  if (!cur) return;
  const r = currentResult();
  if (state.mode === 'undistorted' && r) drawUndistorted(cur, r);
  else drawRaw(cur, r);
  drawCaption(cur, r);
}

function drawRaw(cur, r) {
  const { view } = cur;
  const W = view.width, H = view.height;
  canvas.width = W;
  canvas.height = H;
  ctx.putImageData(new ImageData(toRGBA(view.gray).data, W, H), 0, 0);

  if (state.showLines) drawHeldOutLines(cur, r);
  if (state.showField && r) drawDistortionField(cur, r);
  if (state.showMapValid && r) drawMapValidity(cur, r);
  if (state.showWindow && r) drawSafeWindow(cur, r);
  if (state.showCorners) drawCorners(cur);
  if (state.mode === 'difference' && r) drawCorrectionVectors(cur, r);
}

function drawUndistorted(cur, r) {
  const { image } = getResample(cur, r);
  canvas.width = image.width;
  canvas.height = image.height;
  drawAlphaChecker(image.width, image.height);
  ctx.putImageData(new ImageData(image.data, image.width, image.height), 0, 0);

  if (state.showWindow) {
    ctx.save();
    ctx.strokeStyle = 'rgba(78,161,255,0.9)';
    ctx.lineWidth = Math.max(1, image.width / 500);
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(0.5, 0.5, image.width - 1, image.height - 1);
    ctx.restore();
  }
  if (state.showCorners) drawUndistortedCorners(cur, image);
}

function drawAlphaChecker(w, h) {
  const cell = 12;
  ctx.save();
  for (let y = 0; y < h; y += cell) {
    for (let x = 0; x < w; x += cell) {
      ctx.fillStyle = ((x / cell + y / cell) & 1) ? '#1b212b' : '#141920';
      ctx.fillRect(x, y, cell, cell);
    }
  }
  ctx.restore();
}

function drawCorners(cur) {
  const { view } = cur;
  if (!view.grid) return;
  ctx.save();
  ctx.lineWidth = Math.max(1, view.width / 480);
  const rad = Math.max(2.5, view.width / 220);
  for (let k = 0; k < view.grid.points.length; k++) {
    const [u, v] = view.grid.points[k];
    ctx.beginPath();
    ctx.arc(u, v, rad, 0, Math.PI * 2);
    if (view.excluded.has(k)) {
      ctx.strokeStyle = '#f0b429';
      ctx.stroke();
    } else {
      ctx.fillStyle = 'rgba(74,217,145,0.85)';
      ctx.fill();
    }
  }
  ctx.restore();
}

function drawUndistortedCorners(cur, image) {
  const { view } = cur;
  const r = currentResult();
  if (!view.grid || !r) return;
  const rad = Math.max(2, image.width / 260);
  for (let k = 0; k < view.grid.points.length; k++) {
    const [u, v] = view.grid.points[k];
    const p = undistortPixel(u, v, r.sensor, r.distortionMM);
    if (!p.converged) continue;
    const ox = p.u - image.x0;
    const oy = p.v - image.y0;
    if (ox < 0 || oy < 0 || ox >= image.width || oy >= image.height) continue;
    ctx.beginPath();
    ctx.arc(ox, oy, rad, 0, Math.PI * 2);
    if (view.excluded.has(k)) { ctx.strokeStyle = '#f0b429'; ctx.stroke(); }
    else { ctx.fillStyle = 'rgba(74,217,145,0.9)'; ctx.fill(); }
  }
}

// Where each detected corner moves under the correction.
function drawCorrectionVectors(cur, r) {
  const { view } = cur;
  if (!view.grid) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,107,107,0.8)';
  ctx.lineWidth = Math.max(1, view.width / 480);
  for (const [u, v] of view.grid.points) {
    const p = undistortPixel(u, v, r.sensor, r.distortionMM);
    if (!p.converged) continue;
    ctx.beginPath();
    ctx.moveTo(u, v);
    ctx.lineTo(p.u, p.v);
    ctx.stroke();
  }
  ctx.restore();
}

// Held-out geometry: draw ideal straight lines where they actually land in the
// raw frame. Synthetic uses the true lens; real uses the fitted model.
function drawHeldOutLines(cur, r) {
  if (!r) return;
  const { view, seg } = cur;
  const lines = defaultIdealLines(view.width, view.height, 5);
  ctx.save();
  ctx.lineWidth = Math.max(1, view.width / 480);
  ctx.strokeStyle = 'rgba(255,107,107,0.85)';
  for (const line of lines) {
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < 64; i++) {
      const t = i / 63;
      const iu = line.p0[0] + t * (line.p1[0] - line.p0[0]);
      const iv = line.p0[1] + t * (line.p1[1] - line.p0[1]);
      let ou, ov;
      if (state.dataset.kind === 'synthetic') {
        const obs = truthObservedPixel(iu, iv, truthCameraFor(seg, view));
        ou = obs.u; ov = obs.v;
      } else {
        const xn = (sensorXFromPixel(iu, r.sensor) - r.sensor.px_mm) / r.sensor.fx_mm;
        const yn = (sensorYFromPixel(iv, r.sensor) - r.sensor.py_mm) / r.sensor.fy_mm;
        const src = normalizedToPixel(xn, yn, r.sensor, r.distortionMM);
        ou = src.u; ov = src.v;
      }
      if (ou < 0 || ov < 0 || ou > view.width - 1 || ov > view.height - 1) { started = false; continue; }
      if (!started) { ctx.moveTo(ou, ov); started = true; } else { ctx.lineTo(ou, ov); }
    }
    ctx.stroke();
  }
  ctx.restore();
}

function truthCameraFor(seg, view) {
  const t = state.dataset.truth;
  return {
    width: view.width, height: view.height,
    pitchX: t.pitchX, pitchY: t.pitchY,
    fx_mm: seg.f_mm, fy_mm: seg.f_mm,
    px_mm: t.principalMM.px, py_mm: t.principalMM.py,
    originX_mm: seg.originX_mm ?? 0, originY_mm: seg.originY_mm ?? 0,
    distortion: t.distortion,
  };
}

function drawSafeWindow(cur, r) {
  const d = r.decision;
  const { view, seg } = cur;
  ctx.save();

  // Ghost of the previous solve's window, so even a few-pixel change is visible.
  const prev = state.prevWindows.get(seg.focalId);
  if (prev && d.accepted) {
    ctx.strokeStyle = 'rgba(160,170,185,0.65)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 4]);
    ctx.strokeRect(prev.x0, prev.y0, prev.width, prev.height);
  }

  ctx.lineWidth = Math.max(1.5, view.width / 320);
  if (!d.accepted) {
    ctx.strokeStyle = 'rgba(255,107,107,0.95)';
    ctx.fillStyle = 'rgba(255,107,107,0.10)';
    ctx.setLineDash([10, 6]);
    ctx.fillRect(1, 1, view.width - 2, view.height - 2);
    ctx.strokeRect(1, 1, view.width - 2, view.height - 2);
  } else {
    const w = d.window;
    ctx.strokeStyle = d.shrunk ? 'rgba(240,180,41,0.95)' : 'rgba(78,161,255,0.9)';
    ctx.fillStyle = d.shrunk ? 'rgba(240,180,41,0.07)' : 'rgba(78,161,255,0.05)';
    ctx.setLineDash(d.shrunk ? [8, 5] : []);
    ctx.fillRect(w.x0, w.y0, w.width, w.height);
    ctx.strokeRect(w.x0 + 0.5, w.y0 + 0.5, w.width - 1, w.height - 1);
  }
  ctx.restore();
}

function drawDistortionField(cur, r) {
  const { view } = cur;
  const n = 17;
  ctx.save();
  ctx.strokeStyle = 'rgba(78,161,255,0.75)';
  ctx.lineWidth = 1;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = (i / (n - 1)) * (view.width - 1);
      const v = (j / (n - 1)) * (view.height - 1);
      const f = distortionFieldAt(u, v, r.sensor, r.distortionMM);
      if (!f.converged || f.fold) continue;
      const xn = (sensorXFromPixel(u, r.sensor) - r.sensor.px_mm) / r.sensor.fx_mm;
      const yn = (sensorYFromPixel(v, r.sensor) - r.sensor.py_mm) / r.sensor.fy_mm;
      const src = normalizedToPixel(xn, yn, r.sensor, r.distortionMM);
      const scale = 8;
      ctx.beginPath();
      ctx.moveTo(u, v);
      ctx.lineTo(u + (src.u - u) * scale, v + (src.v - v) * scale);
      ctx.stroke();
      ctx.fillStyle = 'rgba(78,161,255,0.9)';
      ctx.fillRect(u - 0.5, v - 0.5, 1.5, 1.5);
    }
  }
  ctx.restore();
}

function drawMapValidity(cur, r) {
  const map = r.map;
  ctx.save();
  for (let j = 0; j < map.height; j++) {
    for (let i = 0; i < map.width; i++) {
      if (map.valid[j * map.width + i]) continue;
      ctx.fillStyle = 'rgba(255,107,107,0.35)';
      ctx.fillRect(map.originX + i * map.step, map.originY + j * map.step, map.step, map.step);
    }
  }
  ctx.restore();
}

function drawCaption(cur, r) {
  const { seg, view } = cur;
  const parts = [`${seg.focalId} · ${view.id}`, `${view.width}×${view.height}`];
  if (state.mode === 'undistorted' && r?.decision?.accepted) {
    const w = r.decision.window;
    parts.push(`output ${Math.round(w.width)}×${Math.round(w.height)} @ ${Math.round(w.x0)},${Math.round(w.y0)}`);
  }
  $('canvas-caption').textContent = parts.join('  ·  ');
}

function getResample(cur, r) {
  const w = r.decision.window;
  const key = `${cur.view.id}|${state.path}|${r.model}|${r.rms.toFixed(5)}|${w ? w.width : 'rej'}|${w ? w.height : ''}`;
  let hit = state.resampleCache.get(key);
  if (!hit) {
    hit = resampleView(cur.view, r);
    state.resampleCache.set(key, hit);
    if (state.resampleCache.size > 24) {
      state.resampleCache.delete(state.resampleCache.keys().next().value);
    }
  }
  return hit;
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function renderViewList() {
  const list = $('view-list');
  list.innerHTML = '';
  let total = 0, used = 0;
  for (const seg of state.dataset.segments) {
    for (const view of seg.views) {
      total++;
      if (view.grid && !view.deleted) used++;
    }
  }
  $('views-summary').textContent = `${used}/${total} used`;

  for (const seg of state.dataset.segments) {
    for (const view of seg.views) {
      const held = state.heldOut.has(view.id);
      const item = document.createElement('div');
      item.className = 'view-item'
        + (view.id === state.viewId ? ' active' : '')
        + (view.deleted ? ' deleted' : '');

      const title = document.createElement('div');
      title.innerHTML = `<div class="vi-title">${view.id}${held ? ' <span class="muted">(held out)</span>' : ''}</div>
        <div class="vi-sub">${view.grid
          ? `${view.grid.points.length} corners${view.grid.partial ? ' · partial' : ''}${view.excluded.size ? ` · ${view.excluded.size} excluded` : ''}`
          : 'no board'}</div>`;

      const holdBtn = document.createElement('button');
      holdBtn.textContent = held ? 'fit' : 'hold';
      holdBtn.title = 'Toggle held out of the fit';
      holdBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (held) state.heldOut.delete(view.id); else state.heldOut.add(view.id);
        renderViewList();
        scheduleSolve();
      });

      const delBtn = document.createElement('button');
      delBtn.textContent = view.deleted ? 'restore' : 'delete';
      delBtn.title = 'Delete/restore this view from the fit';
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        view.deleted = !view.deleted;
        renderViewList();
        scheduleSolve();
      });

      item.addEventListener('click', () => {
        state.viewId = view.id;
        $('view-select').value = view.id;
        renderViewList();
        draw();
      });

      item.appendChild(title);
      item.appendChild(holdBtn);
      item.appendChild(delBtn);
      list.appendChild(item);
    }
  }
}

function renderFit() {
  const el = $('fit-out');
  const r = currentResult();
  if (!state.result?.ok || !r) {
    const msg = state.result?.errors?.[0] ?? 'not solved';
    el.innerHTML = `<div class="k">status</div><div class="v bad">${msg}</div>`;
    return;
  }
  const K = r.K;
  const rows = [
    ['path', state.path === 'shared' ? 'shared sensor' : 'per focal'],
    ['model used', r.model],
    ['focal', `${r.sensor.fx_mm.toFixed(2)} mm`],
    ['fx / fy px', `${K.fx.toFixed(1)} / ${K.fy.toFixed(1)}`],
    ['principal px', `${K.cx.toFixed(1)}, ${K.cy.toFixed(1)}`],
    ['principal mm', `${r.sensor.px_mm.toFixed(3)}, ${r.sensor.py_mm.toFixed(3)}`],
    ['reproj RMS', `${r.rms.toFixed(3)} px`],
    ['converged', r.converged ? 'yes' : 'no'],
    ['k1 (mm)', r.distortionMM.k1.toExponential(3)],
    ['k2 (mm)', r.distortionMM.k2.toExponential(3)],
    ['k3 (mm)', r.distortionMM.k3.toExponential(3)],
    ['p1, p2 (mm)', `${r.distortionMM.p1.toExponential(2)}, ${r.distortionMM.p2.toExponential(2)}`],
  ];
  let html = rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join('');
  if (r.note) html += `<div class="k">note</div><div class="v warn">${r.note}</div>`;
  el.innerHTML = html;
}

function renderWindow() {
  const el = $('window-out');
  const note = $('window-note');
  const cur = currentView();
  const r = currentResult();
  if (!r) { el.innerHTML = ''; note.textContent = ''; return; }
  const d = r.decision;
  const m = r.map;
  const rows = [
    ['map samples', `${m.stats.valid} / ${m.stats.total} usable`],
    ['folds (det<=0)', String(m.stats.folds)],
    ['outside source', String(m.stats.outside)],
    ['min Jacobian', m.stats.minJacobianDet.toExponential(2)],
    ['sign flips', String(r.safety.foldFlips)],
    ['jumps', String(r.safety.discontinuities)],
  ];
  if (d.accepted) {
    rows.push(['decision', d.shrunk ? 'shrunk' : 'accepted']);
    rows.push(['window px', `${Math.round(d.window.width)} × ${Math.round(d.window.height)}`]);
    rows.push(['origin px', `${Math.round(d.window.x0)}, ${Math.round(d.window.y0)}`]);
    const prev = state.prevWindows.get(cur?.seg.focalId);
    if (prev && prev.width !== d.window.width) {
      rows.push(['Δ vs previous', `${Math.round(d.window.width - prev.width)} × ${Math.round(d.window.height - prev.height)} px`]);
    }
  } else {
    rows.push(['decision', 'REJECTED']);
  }
  el.innerHTML = rows.map(([k, v]) => {
    let cls = '';
    if (k === 'decision') cls = d.accepted ? (d.shrunk ? 'warn' : 'good') : 'bad';
    return `<div class="k">${k}</div><div class="v ${cls}">${v}</div>`;
  }).join('');

  if (!d.accepted) {
    note.className = 'note bad';
    note.textContent = `${d.reason}. This segment is not exported; no average residual can make its map samplable.`;
  } else if (d.shrunk) {
    note.className = 'note warn';
    note.textContent = 'The requested frame is not fully covered by a fold-free map, so the crop above is the largest safe sub-window of it.';
  } else {
    note.className = 'note';
    note.textContent = 'The whole request is fold-free and inside source coverage.';
  }
}

function renderVerify() {
  const el = $('verify-out');
  const note = $('verify-note');
  const r = currentResult();
  if (!r?.verification) { el.innerHTML = ''; note.textContent = ''; return; }
  const v = r.verification;
  const rows = [];
  if (v.kind === 'synthetic') {
    rows.push(['center offset', `${v.center.distanceMM.toFixed(3)} mm / ${v.center.distancePx.toFixed(1)} px`]);
    rows.push(['periphery fail', `${(r.invHealth.failureFraction * 100).toFixed(2)}%`]);
    rows.push(['edge folds', String(r.invHealth.edgeFolds)]);
    rows.push(['held-out bend', `${v.lineBending.maxDeviationPx.toFixed(3)} px max`]);
    rows.push(['resample RMS', `${v.resampling.rmsPx.toFixed(3)} px`]);
    rows.push(['resample max', `${v.resampling.maxPx.toFixed(3)} px`]);
    note.className = 'note';
    note.textContent = 'Held-out lines were rendered through the true lens and never fitted. Resampling error compares the model map with the true lens over the accepted window.';
  } else {
    if (v.gridBending) rows.push(['lattice bend', `${v.gridBending.maxDeviationPx.toFixed(3)} px max`]);
    rows.push(['held-out views', String(v.heldOutViews ?? 0)]);
    if (v.heldOutCorner) rows.push(['held-out corner', `${v.heldOutCorner.meanPx.toFixed(3)} px mean`]);
    rows.push(['periphery fail', `${(r.invHealth.failureFraction * 100).toFixed(2)}%`]);
    note.className = 'note';
    note.textContent = 'Real photographs have no ground-truth camera, so held-out views and board-lattice straightness are reported instead.';
  }
  el.innerHTML = rows.map(([k, val]) => `<div class="k">${k}</div><div class="v">${val}</div>`).join('');
}

function renderResiduals() {
  const el = $('residuals-out');
  const cur = currentView();
  const r = currentResult();
  if (!r || !cur) { el.innerHTML = ''; return; }
  const used = cur.seg.views.filter((v) => !v.deleted && !state.heldOut.has(v.id) && v.grid);
  const rows = [];
  for (let i = 0; i < Math.min(used.length, r.perView?.length ?? 0); i++) {
    rows.push(`<div class="k">${used[i].id}</div><div class="v">${r.perView[i].rms.toFixed(3)} px</div>`);
  }
  el.innerHTML = rows.join('') || '<div class="k">—</div><div class="v"></div>';
}

// ---------------------------------------------------------------------------
// Solve orchestration
// ---------------------------------------------------------------------------

let solveTimer = null;
let solveQueued = false;
function scheduleSolve() {
  clearTimeout(solveTimer);
  solveTimer = setTimeout(runSolve, 80);
}

async function runSolve() {
  if (!state.dataset) return;
  if (state.busy) {
    // A solve is in flight; queue exactly one more so the latest UI state wins
    // instead of silently dropping the request.
    solveQueued = true;
    return;
  }
  state.busy = true;
  solveQueued = false;
  setStatus('solving…', 'busy');
  $('btn-solve').disabled = true;
  await new Promise((res) => requestAnimationFrame(res));
  try {
    const fitViewsBySeg = new Map();
    for (const seg of state.dataset.segments) {
      fitViewsBySeg.set(seg.focalId,
        seg.views.filter((v) => !v.deleted && !state.heldOut.has(v.id) && v.grid));
    }
    const t0 = performance.now();
    const result = solve(state.dataset, {
      path: state.path,
      model: state.model,
      windowMode: state.windowMode,
      fitViewsBySeg,
    });
    const dt = performance.now() - t0;

    // Update the ghost windows only after a successful solve.
    if (state.result?.perSegment) {
      for (let i = 0; i < state.dataset.segments.length; i++) {
        const prev = state.result.perSegment[i];
        if (prev?.decision?.accepted) {
          state.prevWindows.set(state.dataset.segments[i].focalId, prev.decision.window);
        }
      }
    }

    state.result = result;
    state.solveId++;
    state.resampleCache.clear();
    setStatus(result.ok ? `solved in ${Math.round(dt)} ms` : 'solve failed', result.ok ? 'ok' : 'err');
    $('status').dataset.solves = String(state.solveId);
  } catch (e) {
    console.error(e);
    setStatus('solve error: ' + e.message, 'err');
  } finally {
    state.busy = false;
    $('btn-solve').disabled = false;
  }
  const r = currentResult();
  $('btn-export-map').disabled = !r?.decision?.accepted;
  $('btn-export-json').disabled = !state.result?.ok;
  renderAll();
  if (solveQueued) {
    solveQueued = false;
    scheduleSolve();
  }
}

function renderAll() {
  renderViewList();
  renderFit();
  renderWindow();
  renderVerify();
  renderResiduals();
  draw();
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

canvas.addEventListener('mousemove', (e) => {
  const cur = currentView();
  const r = currentResult();
  if (!cur || state.mode === 'undistorted') { $('hover').textContent = ''; return; }
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
  const parts = [`raw ${x.toFixed(1)}, ${y.toFixed(1)}`];
  if (r) {
    const s = sourceAt(r.map, x, y);
    if (s) parts.push(s.valid ? `map → src ${s.u.toFixed(1)}, ${s.v.toFixed(1)}` : 'map: unusable');
    const f = distortionFieldAt(x, y, r.sensor, r.distortionMM);
    if (!f.converged) parts.push('inverse: no convergence');
    else if (f.fold) parts.push('inverse: FOLD');
    else parts.push(`round-trip ${f.roundTripErrPx.toFixed(3)} px`);
  }
  $('hover').textContent = parts.join('   ·   ');
});

canvas.addEventListener('mouseleave', () => { $('hover').textContent = ''; });

canvas.addEventListener('click', (e) => {
  const cur = currentView();
  if (!cur || state.mode === 'undistorted') return;
  const { view } = cur;
  if (!view.grid) return;
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
  let bestK = -1, bestD = Infinity;
  for (let k = 0; k < view.grid.points.length; k++) {
    const d = Math.hypot(view.grid.points[k][0] - x, view.grid.points[k][1] - y);
    if (d < bestD) { bestD = d; bestK = k; }
  }
  if (bestK < 0 || bestD > 12) return;
  if (view.excluded.has(bestK)) view.excluded.delete(bestK);
  else view.excluded.add(bestK);
  renderViewList();
  scheduleSolve();
});

function bindControls() {
  document.querySelectorAll('input[name="dataset"]').forEach((el) => {
    el.addEventListener('change', () => { if (el.checked) loadDataset(el.value); });
  });
  $('path-select').addEventListener('change', (e) => { state.path = e.target.value; scheduleSolve(); });
  $('model-select').addEventListener('change', (e) => { state.model = e.target.value; scheduleSolve(); });
  $('window-mode').addEventListener('change', (e) => { state.windowMode = e.target.value; scheduleSolve(); });
  $('view-select').addEventListener('change', (e) => {
    state.viewId = e.target.value;
    renderViewList();
    draw();
  });
  $('canvas-mode').addEventListener('change', (e) => { state.mode = e.target.value; draw(); });
  for (const [id, key] of [
    ['show-corners', 'showCorners'], ['show-lines', 'showLines'], ['show-field', 'showField'],
    ['show-window', 'showWindow'], ['show-map-valid', 'showMapValid'],
  ]) {
    $(id).addEventListener('change', (e) => { state[key] = e.target.checked; draw(); });
  }
  $('btn-solve').addEventListener('click', runSolve);
  $('btn-reset-corners').addEventListener('click', () => {
    for (const seg of state.dataset.segments) for (const v of seg.views) v.excluded.clear();
    renderViewList();
    scheduleSolve();
  });
  $('btn-export-map').addEventListener('click', exportMap);
  $('btn-export-json').addEventListener('click', exportReport);
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
}

// The exported map is the same full-resolution destination->source array the
// resampler just consumed, over the accepted window.
function exportMap() {
  const cur = currentView();
  const r = currentResult();
  if (!cur || !r?.decision?.accepted) return;
  const { image, map } = getResample(cur, r);

  const png = encodePng(image);
  download(new Blob([png], { type: 'image/png' }), `undistorted-${cur.view.id}.png`);

  const json = mapToJSON(map, {
    segment: cur.seg.focalId,
    view: cur.view.id,
    path: state.path,
    model: r.model,
    focalMM: r.sensor.fx_mm,
    sensor: {
      width: r.sensor.width, height: r.sensor.height,
      pitchX: r.sensor.pitchX, pitchY: r.sensor.pitchY,
      fx_mm: r.sensor.fx_mm, fy_mm: r.sensor.fy_mm,
      px_mm: r.sensor.px_mm, py_mm: r.sensor.py_mm,
      originX_mm: r.sensor.originX_mm, originY_mm: r.sensor.originY_mm,
    },
    distortionMM: r.distortionMM,
    window: r.decision.window,
  });
  download(new Blob([JSON.stringify(json)], { type: 'application/json' }),
    `remap-${cur.view.id}.json`);
}

function exportReport() {
  const cur = currentView();
  const r = currentResult();
  if (!r) return;
  const report = {
    generatedAt: new Date().toISOString(),
    dataset: state.dataset.kind,
    path: state.path,
    model: r.model,
    view: cur?.view.id,
    fit: {
      focalMM: r.sensor.fx_mm,
      principalMM: { px_mm: r.sensor.px_mm, py_mm: r.sensor.py_mm },
      principalPx: { cx: r.K.cx, cy: r.K.cy },
      distortionMM: r.distortionMM,
      distortionPx: r.distortionPx,
      rmsPx: r.rms,
      converged: r.converged,
    },
    map: {
      stats: r.map.stats,
      safety: r.safety,
      decision: r.decision,
      inverseHealth: r.invHealth,
    },
    verification: r.verification,
    truth: state.dataset.kind === 'synthetic'
      ? { principalMM: state.dataset.truth.principalMM, distortion: state.dataset.truth.distortion }
      : null,
  };
  download(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }),
    `lens-report-${state.dataset.kind}.json`);
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

async function loadDataset(kind) {
  setStatus('loading dataset…', 'busy');
  try {
    state.dataset = kind === 'synthetic' ? loadSynthetic() : await loadReal();
    state.heldOut.clear();
    state.prevWindows.clear();
    state.resampleCache.clear();
    state.result = null;

    const firstWithBoard = state.dataset.segments.flatMap((s) => s.views).find((v) => v.grid);
    state.viewId = firstWithBoard?.id ?? state.dataset.segments[0].views[0].id;

    if (kind === 'real') {
      const usable = state.dataset.segments[0].views.filter((v) => v.grid);
      const { heldOut } = splitHeldOut(usable, 3);
      for (const hv of heldOut) state.heldOut.add(hv.id);
    }

    buildViewSelect();
    $('dataset-hint').textContent = kind === 'synthetic'
      ? 'Zoom + crop sequence with a known lens. Only wide#4 reaches the sensor corners: on the per-focal path, delete it and watch the wide crop shrink. The shared path stays anchored.'
      : 'Real chessboard photographs (OpenCV samples, Apache-2.0). No truth camera; held-out views and lattice straightness are used instead.';
    renderViewList();
    await runSolve();
  } catch (e) {
    console.error(e);
    setStatus('load error: ' + e.message, 'err');
  }
}

function buildViewSelect() {
  const sel = $('view-select');
  sel.innerHTML = '';
  for (const seg of state.dataset.segments) {
    for (const view of seg.views) {
      const opt = document.createElement('option');
      opt.value = view.id;
      opt.textContent = view.id;
      sel.appendChild(opt);
    }
  }
  sel.value = state.viewId;
}

// ---------------------------------------------------------------------------

async function main() {
  bindControls();
  await loadDataset('synthetic');
}

main();
