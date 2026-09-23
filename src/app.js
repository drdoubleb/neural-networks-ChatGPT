(function () {
  'use strict';
  const C = globalThis.NucleusCore, data = globalThis.NUCLEI_DATA;
  const $ = id => document.getElementById(id);
  const trainBank = data.samples.filter(s => s.split === 'train');
  const testBank = data.samples.filter(s => s.split === 'test');
  const rates = [0.001, 0.01, 0.03, 0.08, 0.2, 0.5, 1, 3];
  const pct = v => v == null ? '—' : `${(v * 100).toFixed(0)}%`;
  const signed = (v, digits = 3) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`;
  const color = v => v >= 0 ? '#2b9985' : '#cb7967';
  const state = { stage: 'train', mode: 'shape', hidden: 0, seed: 7, count: 80, rate: 0.08, batch: 8, l2: 0.001, running: false, epoch: 0, updates: 0, cursor: 0, selected: 'N001', neuron: 0, input: 0, imageView: 'contour', shift: 0, stain: 1, version: 0, testRuns: 0, testResults: null, occlusion: null, trace: null, stopAt: 200 };
  let model, scaler, prepared, order, random, history, trainPredictions;
  let frame = 0, lastTick = 0, toastTimer;
  const selectedSample = () => data.samples.find(s => s.id === state.selected) || trainBank[0];
  const threshold = () => +$('threshold').value;
  function toast(message) {
    $('toast').textContent = message; $('toast').classList.add('visible'); clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 3800);
  }
  function currentPixels() { return C.transform(selectedSample().pixels, state.shift, state.stain); }
  function invalidateTest() {
    if (state.testResults) { state.testResults = null; $('test-ready-text').textContent = 'The weights changed after the last evaluation. Run the test again to evaluate the current model.'; }
  }
  function evaluateTrain() {
    trainPredictions = prepared.map(s => { const f = model.forward(s.x); return { ...s, p: f.p, loss: C.lossFromLogit(f.z, s.y) }; });
    return { loss: trainPredictions.reduce((v, s) => v + s.loss, 0) / prepared.length, accuracy: C.metrics(trainPredictions, threshold()).accuracy };
  }
  function reset(announce = true) {
    pause();
    state.mode = $('input-mode').value; state.hidden = +$('hidden-units').value;
    state.count = +$('train-count').value;
    state.seed = Math.max(1, Math.min(99999, Math.round(+$('seed').value || 7))); $('seed').value = state.seed;
    state.epoch = 0; state.updates = 0; state.cursor = 0; state.neuron = 0; state.input = state.mode === 'shape' ? 0 : 119;
    state.shift = 0; state.stain = 1; state.version++; state.trace = null; state.occlusion = null; state.stopAt = 200;
    invalidateTest();
    const active = trainBank.slice(0, state.count);
    scaler = C.fitScaler(active, state.mode);
    prepared = active.map(s => ({ id: s.id, split: s.split, x: C.encode(s.pixels, state.mode, scaler), y: s.label }));
    model = new C.Network(state.mode === 'shape' ? 4 : 256, state.hidden, state.seed);
    random = C.rng(state.seed + 103); order = C.shuffle(prepared, random);
    if (!active.some(s => s.id === state.selected)) state.selected = active[0].id;
    history = [{ epoch: 0, ...evaluateTrain() }];
    $('inspect-neuron').innerHTML = state.hidden ? Array.from({ length: state.hidden }, (_, j) => `<option value="${j}">Hidden ${j + 1}</option>`).join('') : '<option value="0">Output</option>';
    $('input-help').textContent = state.mode === 'shape' ? 'We measure the shape first. The neuron learns how to combine those measurements.' : 'Each 16 × 16 image becomes 256 intensities. No handcrafted measurements enter this model.';
    $('count-value').textContent = `${state.count} / 80`;
    $('follow').checked = true; $('pixel-index').value = state.input + 1;
    syncStress(); syncStage(); render();
    if (announce) toast('Fresh weights. Training and evaluation results have been reset.');
  }
  function pause() {
    state.running = false; cancelAnimationFrame(frame); frame = 0;
    $('play').innerHTML = '<span aria-hidden="true">▶</span> ' + (state.updates ? 'Continue training' : 'Start training');
    $('status').classList.remove('running');
    $('status').innerHTML = '<span class="live-dot"></span>' + (state.updates ? 'Paused' : 'Ready to learn');
  }
  function stepBatch() {
    invalidateTest();
    const batch = order.slice(state.cursor, state.cursor + Math.min(state.batch, prepared.length));
    const first = batch[0], before = model.forward(first.x);
    const index = state.neuron * model.inputs + state.input, old = model.w1[index];
    const result = model.trainBatch(batch, state.rate, state.l2);
    state.trace = { sample: first.id, y: first.y, p: before.p, loss: C.lossFromLogit(before.z, first.y), batch: batch.length, grad: result.g.w1[index], before: old, after: model.w1[index], rate: state.rate, input: state.input, neuron: state.neuron, hidden: state.hidden, mode: state.mode };
    state.cursor += batch.length; state.updates++; state.version++; state.occlusion = null;
    if ($('follow').checked) { state.selected = first.id; state.shift = 0; state.stain = 1; syncStress(); }
    if (state.cursor >= prepared.length) {
      state.epoch++; state.cursor = 0;
      history.push({ epoch: state.epoch, ...evaluateTrain() });
      order = C.shuffle(prepared, random);
    }
  }
  function safeSteps(n) {
    try { for (let i = 0; i < n; i++) stepBatch(); }
    catch (e) { pause(); toast(e.message); return false; }
    evaluateTrain(); render(); return true;
  }
  function tick(now) {
    if (!state.running) return;
    const speed = $('speed').value, delay = speed === 'slow' ? 650 : speed === 'normal' ? 110 : 55;
    if (now - lastTick >= delay) {
      lastTick = now;
      const remaining = Math.ceil((prepared.length - state.cursor) / state.batch) + (state.stopAt - state.epoch - 1) * Math.ceil(prepared.length / state.batch);
      if (!safeSteps(Math.min(speed === 'fast' ? 8 : 1, Math.max(1, remaining)))) return;
      if (state.epoch >= state.stopAt) { pause(); render(); toast(`Paused after ${state.epoch} epochs. Inspect the model or try the test set.`); return; }
    }
    frame = requestAnimationFrame(tick);
  }
  function play() {
    if (state.running) { pause(); render(); return; }
    if (state.epoch >= state.stopAt) state.stopAt = state.epoch + 200;
    state.running = true; lastTick = 0;
    $('play').innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause training';
    $('status').classList.add('running'); $('status').innerHTML = '<span class="live-dot"></span>Learning in progress';
    frame = requestAnimationFrame(tick);
  }
  function syncStage() {
    const isTest = state.stage === 'test', isData = state.stage === 'data';
    document.querySelectorAll('.step-tab').forEach(b => { const active = b.dataset.stage === state.stage; b.classList.toggle('active', active); if (active) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
    $('data-view').hidden = !isData; $('experiment-view').hidden = isData;
    $('controls').hidden = isTest; $('test-overview').hidden = !isTest;
    $('workspace-title').textContent = isTest ? 'Inspect the frozen model.' : 'Watch the connections change.';
    $('workspace').hidden = isTest && !state.testResults;
    $('lab-layout').classList.toggle('test-mode', isTest);
    $('test-empty').hidden = Boolean(state.testResults); $('test-results').hidden = !state.testResults;
    $('learning-panel').hidden = isTest; $('test-influence-panel').hidden = !isTest;
    $('follow-wrap').hidden = isTest; $('next-stage').hidden = isTest;
    $('gallery-title').innerHTML = (isTest ? 'The test bench' : 'The training bench') + ' <span class="muted">/ click a nucleus</span>';
    $('run-test').textContent = state.testResults ? 'Run test again ↻' : 'Run on 20 test nuclei →';
  }
  function setStage(stage) {
    pause(); state.stage = stage; state.shift = 0; state.stain = 1; state.occlusion = null;
    if (stage === 'test' && state.testResults) state.selected = testBank[0].id;
    else if (selectedSample().split !== 'train') state.selected = trainBank[0].id;
    $('gallery-filter').value = 'all'; syncStress(); syncStage(); render();
  }
  function runTest() {
    pause();
    state.testRuns++;
    state.testResults = { epoch: state.epoch, updates: state.updates, version: state.version, predictions: testBank.map(s => { const f = model.forward(C.encode(s.pixels, state.mode, scaler)); return { id: s.id, y: s.label, p: f.p, loss: C.lossFromLogit(f.z, s.label) }; }) };
    state.selected = testBank[0].id; state.shift = 0; state.stain = 1; state.occlusion = null;
    syncStress(); syncStage(); render(); toast('20 predictions complete. No weights were changed.');
  }
  function drawPixels(canvas, pixels, overlay, highlight) {
    const ctx = canvas.getContext('2d'), n = canvas.width, cell = n / 16;
    ctx.clearRect(0, 0, n, n);
    for (let i = 0; i < 256; i++) {
      const v = pixels[i];
      ctx.fillStyle = `rgb(${[247, 241, 250].map((b, c) => Math.round(b + v * ([95, 53, 146][c] - b))).join(',')})`;
      ctx.fillRect(i % 16 * cell, Math.floor(i / 16) * cell, cell + 0.1, cell + 0.1);
    }
    if (overlay) {
      const max = Math.max(0.03, ...overlay.deltas.map(Math.abs));
      overlay.deltas.forEach((delta, k) => {
        if (Math.abs(delta) < 0.00005) return;
        ctx.fillStyle = delta >= 0 ? `rgba(24,165,138,${0.12 + 0.70 * Math.abs(delta) / max})` : `rgba(224,116,85,${0.12 + 0.70 * Math.abs(delta) / max})`;
        ctx.fillRect(k % 8 * cell * 2, Math.floor(k / 8) * cell * 2, cell * 2, cell * 2);
      });
    }
    if (highlight != null) { ctx.strokeStyle = '#193b3b'; ctx.lineWidth = 3; ctx.strokeRect(highlight % 16 * cell + 1.5, Math.floor(highlight / 16) * cell + 1.5, cell - 3, cell - 3); }
  }
  function influence() {
    const key = `${state.version}-${state.selected}-${state.shift}-${state.stain}`;
    if (!state.occlusion || state.occlusion.key !== key) state.occlusion = { key, ...C.occlusion(model, currentPixels(), state.mode, scaler) };
    return state.occlusion;
  }
  function drawSpecimen() {
    const canvas = $('specimen-canvas'), s = selectedSample(), pixels = currentPixels();
    if (state.imageView === 'contour' && !state.shift && state.stain === 1) {
      const ctx = canvas.getContext('2d'), unit = canvas.width / 16;
      ctx.fillStyle = '#f7f1fa'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = `rgb(${[247, 241, 250].map((b, c) => Math.round(b + s.stain * ([95, 53, 146][c] - b))).join(',')})`;
      ctx.beginPath(); s.contour.forEach((p, i) => i ? ctx.lineTo(p[0] * unit, p[1] * unit) : ctx.moveTo(p[0] * unit, p[1] * unit)); ctx.closePath(); ctx.fill();
    } else drawPixels(canvas, pixels, state.imageView === 'influence' ? influence() : null);
    $('image-caption').textContent = state.imageView === 'influence' ? '2 × 2 patch removal. Teal supports irregular; coral opposes it. Scale adapts to this image.' : state.imageView === 'pixels' || state.shift || state.stain !== 1 ? 'The exact 16 × 16 intensity image supplied to the input pipeline.' : 'Smooth source drawing. The model receives its 16 × 16 raster.';
    $('influence-note').hidden = state.imageView !== 'influence';
    if (state.imageView === 'influence') {
      const deltas = influence().deltas, max = Math.max(...deltas.map(Math.abs));
      $('influence-detail').textContent = `Largest patch effect in this image: ${(max * 100).toFixed(1)} percentage points.`;
      readPatch(+$('patch-index').value - 1);
    }
    canvas.setAttribute('aria-label', `${s.id}, ${s.label ? 'irregular' : 'regular'} reference label, ${state.imageView} view`);
  }
  function svgText(x, y, text, size = 11, anchor = 'middle', extra = '') { return `<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" ${extra}>${text}</text>`; }
  function drawNetwork(f) {
    const hidden = state.hidden, pixel = state.mode === 'pixels', svg = $('network-svg');
    const ix = pixel ? 194 : 141, hx = 361, ox = hidden ? 553 : 485;
    const inputs = pixel ? [54, 73, 85, 99, 105, 118, 122, 137, 150, 167, 185, 202] : [0, 1, 2, 3];
    if (pixel && !inputs.includes(state.input)) inputs[11] = state.input;
    const iy = i => pixel ? 76 + Math.floor(i / 16) * 12 : 73 + i * 69;
    const hy = j => hidden === 1 ? 176 : 50 + j * 247 / (hidden - 1);
    const maxWeight = Math.max(0.2, ...model.w1.map(Math.abs), ...model.w2.map(Math.abs));
    let edges = '', nodes = '', labels = '';
    const edge = (x1, y1, x2, y2, w, emphasis = false) => {
      const width = 0.5 + 3 * Math.abs(w) / maxWeight, opacity = emphasis ? 0.95 : pixel || hidden ? 0.22 + 0.33 * Math.abs(w) / maxWeight : 0.75;
      const d = `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`;
      return `<path d="${d}" fill="none" stroke="${color(w)}" stroke-width="${width}" opacity="${opacity}"><title>Weight ${signed(w, 5)}</title></path>` + (state.running && emphasis ? `<path d="${d}" fill="none" stroke="${color(w)}" stroke-width="1.7" class="flow-path"/>` : '');
    };
    for (const i of inputs) {
      for (let j = 0; j < (hidden || 1); j++) {
        const w = model.w1[j * model.inputs + i];
        edges += edge(ix + (pixel ? 0 : 17), iy(i), hidden ? hx - 14 : ox - 31, hidden ? hy(j) : 177, w, j === state.neuron && i === state.input);
        if (!hidden && !pixel) labels += `<rect x="290" y="${(iy(i) + 177) / 2 - 9}" width="66" height="20" rx="4" fill="white"/>` + svgText(323, (iy(i) + 177) / 2 + 5, signed(w, 2), 12, 'middle', `style="fill:${color(w)};font-family:var(--mono)"`);
      }
    }
    if (pixel) {
      nodes += `<rect x="1" y="62" width="194" height="194" rx="5" fill="#f7f1fa" stroke="#e1d9e8"/>`;
      const pixels = currentPixels();
      for (let i = 0; i < 256; i++) nodes += `<rect x="${2 + i % 16 * 12}" y="${63 + Math.floor(i / 16) * 12}" width="12" height="12" fill="#765493" opacity="${pixels[i]}"/>`;
      const r = state.input;
      nodes += `<rect x="${2 + r % 16 * 12}" y="${63 + Math.floor(r / 16) * 12}" width="12" height="12" fill="none" stroke="#193b3b" stroke-width="2"/>`;
      labels += svgText(99, 31, '256 PIXEL INPUTS', 11) + svgText(98, 284, '12 connections per neuron shown', 9) + svgText(98, 300, 'All 256 are used in the calculation.', 9);
    } else {
      labels += svgText(105, 30, '4 MEASURED INPUTS', 11);
      inputs.forEach(i => {
        const y = iy(i), value = f.x[i];
        nodes += `<g class="node-action" tabindex="0" role="button" aria-label="Inspect ${C.FEATURES[i].name} weight" data-input="${i}"><circle cx="${ix}" cy="${y}" r="18" fill="${value >= 0 ? '#e4f0e9' : '#f5e6df'}" stroke="${i === state.input ? '#213a3d' : '#c4d4cb'}" stroke-width="${i === state.input ? 2 : 1}"/><title>${C.FEATURES[i].name}: standardized input ${signed(value, 4)}</title>${svgText(ix, y + 4, signed(value, 1), 10)}</g>`;
        labels += svgText(112, y - 5, C.FEATURES[i].name, 11, 'end') + svgText(112, y + 11, 'standardized', 8, 'end', 'style="fill:#83928c"');
      });
    }
    if (hidden) {
      labels += svgText(hx, 26, `${hidden} HIDDEN · tanh`, 10);
      for (let j = 0; j < hidden; j++) {
        edges += edge(hx + 15, hy(j), ox - 31, 177, model.w2[j], j === state.neuron);
        const a = f.a[j], r = hidden > 8 ? 10 : 14;
        nodes += `<g class="node-action" role="button" tabindex="0" aria-label="Inspect hidden neuron ${j + 1}, activation ${a.toFixed(3)}" data-neuron="${j}"><circle cx="${hx}" cy="${hy(j)}" r="${r}" fill="${color(a)}" fill-opacity="${0.15 + Math.abs(a) * 0.65}" stroke="${j === state.neuron ? '#213a3d' : '#c7d4cc'}" stroke-width="${j === state.neuron ? 2.5 : 1}"/><title>Hidden ${j + 1}: activation ${signed(a, 4)}, bias ${signed(model.b1[j], 4)}, output weight ${signed(model.w2[j], 4)}</title>${svgText(hx, hy(j) + 3, j + 1, 8)}</g>`;
        nodes += svgText(hx + 20, hy(j) + 4, signed(a, 2), 9, 'start');
      }
      labels += svgText(hx, 331, 'Click a neuron to inspect it', 9);
    }
    nodes += `<circle cx="${ox}" cy="177" r="33" fill="#eee8f5" stroke="#9678af" stroke-width="1.5"/>${svgText(ox, 182, pct(f.p), 18, 'middle', 'style="font-family:var(--mono)"')}`;
    labels += svgText(ox, 114, 'OUTPUT', 11) + svgText(ox, 132, 'sigmoid', 10) + svgText(ox, 235, 'P(irregular)', 11) + svgText(ox, 253, `bias ${signed(hidden ? model.b2[0] : model.b1[0], 2)}`, 10);
    if (!hidden) labels += svgText(330, 332, 'weighted sum + bias → sigmoid', 11);
    svg.innerHTML = `<title>${hidden ? `${hidden} hidden tanh neurons and one sigmoid output` : 'One sigmoid neuron'}; actual weights and activations for ${state.selected}</title>${edges}${nodes}${labels}`;
  }
  function renderWeights(f) {
    const j = state.neuron, isPixel = state.mode === 'pixels';
    const weights = model.w1.slice(j * model.inputs, (j + 1) * model.inputs);
    const max = Math.max(0.1, ...weights.map(Math.abs));
    $('weight-title').textContent = state.hidden ? `Hidden neuron ${j + 1}'s weights` : "The output neuron's weights";
    $('weight-description').textContent = isPixel ? 'A 16 × 16 map of input weights, with a symmetric color scale. Inspect any pixel to see its value and contribution.' : 'Four standardized inputs. Bars show the signed weight; the final column shows weight × input.';
    $('shape-weights').hidden = isPixel; $('pixel-weights').hidden = !isPixel;
    if (!isPixel) {
      $('shape-weights').innerHTML = '<div class="weight-row head"><span>INPUT</span><span>WEIGHT</span><span></span><span>W × X</span></div>' + weights.map((w, i) => `<div class="weight-row ${i === state.input ? 'selected' : ''}"><button data-weight-input="${i}" title="${C.FEATURES[i].detail}">${C.FEATURES[i].name}</button><span class="mono">${signed(w, 2)}</span><div class="weight-bar"><i class="${w >= 0 ? 'plus' : 'minus'}" style="width:${Math.abs(w) / max * 50}%"></i></div><span class="mono">${signed(w * f.x[i], 2)}</span></div>`).join('');
    } else {
      const canvas = $('weight-canvas'), ctx = canvas.getContext('2d'), cell = canvas.width / 16;
      ctx.fillStyle = '#fafbf8'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      weights.forEach((w, i) => { ctx.globalAlpha = Math.abs(w) / max; ctx.fillStyle = color(w); ctx.fillRect(i % 16 * cell, Math.floor(i / 16) * cell, cell, cell); });
      ctx.globalAlpha = 1; ctx.strokeStyle = '#193b3b'; ctx.lineWidth = 2; ctx.strokeRect(state.input % 16 * cell + 1, Math.floor(state.input / 16) * cell + 1, cell - 2, cell - 2);
      $('pixel-detail').innerHTML = `row ${Math.floor(state.input / 16) + 1}, col ${state.input % 16 + 1}<br>intensity ${currentPixels()[state.input].toFixed(3)}<br>x = ${signed(f.x[state.input])}<br>w = ${signed(weights[state.input])}<br>w × x = ${signed(weights[state.input] * f.x[state.input])}`;
    }
    $('bias-line').innerHTML = `Bias <b>${signed(model.b1[j])}</b> · pre-activation <b>${signed(f.z1[j])}</b> · ${state.hidden ? 'tanh' : 'sigmoid'} <b>${f.a[j].toFixed(3)}</b>` + (state.hidden ? `<br>Weight to output <b>${signed(model.w2[j])}</b> · contribution to output score <b>${signed(model.w2[j] * f.a[j])}</b>` : '');
    const sum = weights.reduce((s, w, i) => s + w * f.x[i], 0);
    $('calculation').innerHTML = `z${state.hidden ? 'ₕ' : ''} = Σ(w × x) + b<br>= ${signed(sum)} ${signed(model.b1[j])}<br>= ${signed(f.z1[j])}<br>${state.hidden ? `aₕ = tanh(zₕ) = ${f.a[j].toFixed(4)}<br>z_out = Σ(vₕ × aₕ) + b_out<br>= ${signed(f.z)}<br>` : ''}P(irregular) = 1 / (1 + exp(−z${state.hidden ? '_out' : ''}))<br>= ${f.p.toFixed(4)}`;
  }
  function drawChart() {
    const w = 490, h = 190, left = 42, top = 12, right = 12, bottom = 33;
    const plotW = w - left - right, plotH = h - top - bottom;
    const ymax = Math.max(0.8, ...history.map(p => p.loss)) * 1.08, xmax = Math.max(10, state.epoch);
    const x = epoch => left + epoch / xmax * plotW, y = loss => top + plotH - loss / ymax * plotH;
    let svg = '';
    for (let i = 0; i < 4; i++) {
      const v = ymax * i / 3;
      svg += `<line x1="${left}" y1="${y(v)}" x2="${w - right}" y2="${y(v)}" stroke="#e8ede6"/>${svgText(left - 7, y(v) + 3, v.toFixed(2), 9, 'end', 'style="fill:#89948e"')}`;
    }
    for (let i = 0; i < 5; i++) svg += svgText(x(xmax * i / 4), h - 16, Math.round(xmax * i / 4), 9, 'middle', 'style="fill:#89948e"');
    const points = history.map(p => `${x(p.epoch)},${y(p.loss)}`);
    svg += `<path d="M${left},${y(0)} L${points.join(' L')} L${x(history.at(-1).epoch)},${y(0)} Z" fill="#e4f0e9" opacity=".65"/><polyline points="${points.join(' ')}" fill="none" stroke="#217b6d" stroke-width="2.5" stroke-linejoin="round"/>`;
    const last = history.at(-1);
    svg += `<circle cx="${x(last.epoch)}" cy="${y(last.loss)}" r="3.5" fill="#217b6d"/>${svgText(w / 2, h - 1, 'completed epochs', 8, 'middle', 'style="fill:#89948e"')}`;
    if (history.length === 1) svg += svgText(265, 65, 'The curve appears as epochs complete.', 11, 'middle', 'style="fill:#89948e"');
    $('loss-chart').innerHTML = svg;
    $('loss-chart').setAttribute('aria-label', `Training cross-entropy. Initial loss ${history[0].loss.toFixed(3)}. At epoch ${last.epoch}, loss ${last.loss.toFixed(3)}.`);
  }
  function renderTrace() {
    const t = state.trace;
    if (!t) { $('update-trace').innerHTML = '<b>Try “1 update”.</b><p>See one weight change using the real gradient of a batch.</p>'; return; }
    const name = t.mode === 'shape' ? C.FEATURES[t.input].name : `Pixel ${t.input + 1}`;
    $('update-trace').innerHTML = `<b>Last update · ${name} → ${t.hidden ? `hidden ${t.neuron + 1}` : 'output'}</b><p>First example in batch: ${t.sample}, target ${t.y}. Before update, P(irregular) = ${t.p.toFixed(3)}; error p − y = ${signed(t.p - t.y)}.</p><span class="mono">w_new = w_old − learning rate × gradient<br>${signed(t.after, 4)} = ${signed(t.before, 4)} − ${t.rate} × (${signed(t.grad, 4)})</span><p>Gradient averages ${t.batch} example${t.batch === 1 ? '' : 's'} + the L2 term. This is the actual parameter update.</p>`;
  }
  function tile(s, pred, bank = false) {
    const right = pred ? (pred.p >= threshold() ? 1 : 0) === s.label : null;
    const name = s.label ? 'Irregular' : 'Regular';
    return `<button class="nucleus-tile ${pred ? right ? 'right' : 'wrong' : ''} ${!bank && s.id === state.selected ? 'selected' : ''}" data-sample="${s.id}" aria-label="${s.id}, reference ${name}${pred ? `, P irregular ${pct(pred.p)}, ${right ? 'correct' : 'incorrect'}` : ''}" ${!bank ? `aria-pressed="${s.id === state.selected}"` : ''}><img src="nuclei/${s.id}.svg" alt="" loading="lazy">${pred ? `<span class="tile-symbol" aria-hidden="true">${right ? '✓' : '×'}</span>` : ''}<span class="tile-id">${s.id}</span>${pred ? `<span class="tile-prediction">${pct(pred.p)}</span>` : ''}<span class="tile-truth">${name}</span></button>`;
  }
  function renderGallery() {
    const isTest = state.stage === 'test', samples = isTest ? testBank : trainBank.slice(0, state.count);
    const predictions = isTest ? state.testResults?.predictions : trainPredictions;
    if (!predictions) { $('sample-gallery').innerHTML = ''; delete $('sample-gallery').dataset.signature; return; }
    const lookup = new Map(predictions.map(p => [p.id, p])), filter = $('gallery-filter').value;
    const filtered = samples.filter(s => filter === 'all' || filter === 'regular' && s.label === 0 || filter === 'irregular' && s.label === 1 || filter === 'wrong' && +(lookup.get(s.id).p >= threshold()) !== s.label);
    const gallery = $('sample-gallery'), signature = filtered.map(s => s.id).join(',');
    // Keep image elements alive during training: recreating them every frame can
    // refetch all 80 SVGs on a no-cache server and also discards keyboard focus.
    if (gallery.dataset.signature !== signature || gallery.children.length !== filtered.length) {
      gallery.innerHTML = filtered.map(s => tile(s, lookup.get(s.id))).join('');
      gallery.dataset.signature = signature;
    } else {
      [...gallery.children].forEach((button, i) => {
        const s = filtered[i], p = lookup.get(s.id).p, right = +(p >= threshold()) === s.label;
        button.classList.toggle('right', right); button.classList.toggle('wrong', !right); button.classList.toggle('selected', s.id === state.selected);
        button.setAttribute('aria-pressed', s.id === state.selected);
        button.setAttribute('aria-label', `${s.id}, reference ${s.label ? 'Irregular' : 'Regular'}, P irregular ${pct(p)}, ${right ? 'correct' : 'incorrect'}`);
        button.querySelector('.tile-prediction').textContent = pct(p);
        button.querySelector('.tile-symbol').textContent = right ? '✓' : '×';
      });
    }
    $('gallery-empty').hidden = Boolean(filtered.length);
    $('gallery-note').textContent = `Showing ${filtered.length} of ${samples.length} ${isTest ? 'test' : 'active training'} nuclei. ✓ / × = correct / incorrect. Percentage = P(irregular); text = reference label.`;
  }
  function renderTest() {
    if (!state.testResults) return;
    const m = C.metrics(state.testResults.predictions, threshold());
    $('test-accuracy').textContent = pct(m.accuracy); $('test-count').textContent = `${m.tp + m.tn} of ${m.total} correct`;
    $('test-sensitivity').textContent = pct(m.sensitivity); $('test-specificity').textContent = pct(m.specificity);
    $('confusion').innerHTML = `<table><caption>Confusion matrix · counts</caption><thead><tr><th>Reference ↓ / Call →</th><th>Regular</th><th>Irregular</th></tr></thead><tbody><tr><th scope="row">Regular</th><td>${m.tn}</td><td class="error">${m.fp}</td></tr><tr><th scope="row">Irregular</th><td class="error">${m.fn}</td><td>${m.tp}</td></tr></tbody></table>`;
    $('evaluation-note').textContent = `Evaluated at epoch ${state.testResults.epoch}, update ${state.testResults.updates}. Current threshold: ${threshold().toFixed(2)}. Evaluation ${state.testRuns} this session; no test-set weight updates.`;
  }
  function render() {
    if (!model) return;
    const pixels = currentPixels(), f = model.forward(C.encode(pixels, state.mode, scaler)), sample = selectedSample();
    const m = C.metrics(trainPredictions, threshold());
    $('epoch-value').textContent = state.epoch + (state.cursor ? `.${Math.floor(state.cursor / prepared.length * 10)}` : '');
    $('updates-value').textContent = state.updates; $('params-value').textContent = model.parameters().length.toLocaleString();
    $('accuracy-value').textContent = pct(m.accuracy); $('loss-value').textContent = (trainPredictions.reduce((sum, s) => sum + s.loss, 0) / trainPredictions.length).toFixed(3);
    $('sample-id').textContent = sample.id + (state.shift || state.stain !== 1 ? ' *' : '');
    $('truth-label').textContent = sample.label ? 'Irregular' : 'Regular'; $('truth-label').classList.toggle('irregular', Boolean(sample.label));
    $('prediction-value').textContent = `${(f.p * 100).toFixed(1)}%`;
    const call = +(f.p >= threshold()), right = call === sample.label;
    $('prediction-call').textContent = `${call ? 'Irregular' : 'Regular'} ${right ? '✓' : '×'}`;
    $('prediction-call').classList.toggle('incorrect', !right);
    $('probability-fill').style.width = `${f.p * 100}%`; $('threshold-marker').style.left = `${threshold() * 100}%`; $('threshold-value').textContent = threshold().toFixed(2);
    drawSpecimen(); drawNetwork(f); renderWeights(f); drawChart(); renderTrace(); renderGallery(); renderTest();
    if (state.shift || state.stain !== 1) {
      const base = model.forward(C.encode(sample.pixels, state.mode, scaler)).p;
      $('stress-result').textContent = `Original P(irregular) ${pct(base)} → modified ${pct(f.p)} (${signed((f.p - base) * 100, 1)} percentage points). Reported metrics still use the original image.`;
    } else $('stress-result').textContent = 'Try moving the image without changing the underlying class. Does the output stay the same?';
    $('status').innerHTML = '<span class="live-dot"></span>' + (state.stage === 'test' ? 'Fixed weights · inference' : state.running ? 'Learning in progress' : state.updates ? 'Paused' : 'Ready to learn');
  }
  function selectSample(id) {
    state.selected = id; state.shift = 0; state.stain = 1; state.occlusion = null; $('follow').checked = false;
    syncStress(); render();
  }
  function syncStress() {
    $('shift').value = state.shift; $('stain').value = state.stain;
    $('shift-value').textContent = `${state.shift} px`; $('stain-value').textContent = `${state.stain.toFixed(2)}×`;
  }
  function showPatch(event) {
    if (state.imageView !== 'influence') return;
    const rect = $('specimen-canvas').getBoundingClientRect();
    const x = Math.max(0, Math.min(7, Math.floor((event.clientX - rect.left) / rect.width * 8))), y = Math.max(0, Math.min(7, Math.floor((event.clientY - rect.top) / rect.height * 8)));
    $('patch-index').value = y * 8 + x + 1;
    readPatch(y * 8 + x);
  }
  function readPatch(index) {
    const k = Math.max(0, Math.min(63, Math.floor(index) || 0)), x = k % 8, y = Math.floor(k / 8);
    const map = influence(), d = map.deltas[k];
    $('occlusion-readout').textContent = `Patch row ${y + 1}, column ${x + 1}: original ${pct(map.base)} → erased ${pct(map.base - d)}. Difference (original − erased): ${signed(d * 100, 2)} percentage points.`;
  }
  function inspectInput(i) {
    state.input = Math.max(0, Math.min(model.inputs - 1, i)); $('pixel-index').value = state.input + 1;
    render();
  }
  function clearPreset() { document.querySelectorAll('.preset').forEach(b => b.classList.remove('active')); }
  document.querySelectorAll('.step-tab').forEach(b => b.addEventListener('click', () => setStage(b.dataset.stage)));
  $('data-next').addEventListener('click', () => setStage('train')); $('go-test').addEventListener('click', () => { setStage('test'); document.querySelector('.step-nav').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
  $('run-test').addEventListener('click', runTest); $('play').addEventListener('click', play);
  $('step').addEventListener('click', () => { pause(); safeSteps(1); });
  $('epoch-step').addEventListener('click', () => { pause(); safeSteps(Math.ceil((prepared.length - state.cursor) / Math.min(state.batch, prepared.length))); });
  $('reset').addEventListener('click', () => reset());
  ['input-mode', 'hidden-units', 'seed', 'train-count'].forEach(id => $(id).addEventListener('change', () => { clearPreset(); reset(); }));
  $('train-count').addEventListener('input', () => $('count-value').textContent = `${$('train-count').value} / 80`);
  $('learning-rate').addEventListener('input', () => { state.rate = rates[+$('learning-rate').value]; $('rate-value').textContent = state.rate; });
  $('batch-size').addEventListener('change', () => { state.batch = +$('batch-size').value; });
  $('l2').addEventListener('change', () => { state.l2 = +$('l2').value; });
  $('threshold').addEventListener('input', render);
  $('gallery-filter').addEventListener('change', renderGallery);
  $('inspect-neuron').addEventListener('change', () => { state.neuron = +$('inspect-neuron').value; render(); });
  $('pixel-index').addEventListener('change', () => inspectInput((+$('pixel-index').value || 1) - 1));
  $('weight-canvas').addEventListener('click', e => { const r = e.target.getBoundingClientRect(); inspectInput(Math.min(15, Math.floor((e.clientY - r.top) / r.height * 16)) * 16 + Math.min(15, Math.floor((e.clientX - r.left) / r.width * 16))); });
  $('shape-weights').addEventListener('click', e => { const b = e.target.closest('[data-weight-input]'); if (b) inspectInput(+b.dataset.weightInput); });
  function networkClick(e) { const n = e.target.closest('[data-neuron]'), i = e.target.closest('[data-input]'); if (n) { state.neuron = +n.dataset.neuron; $('inspect-neuron').value = state.neuron; render(); } else if (i) inspectInput(+i.dataset.input); }
  $('network-svg').addEventListener('click', networkClick);
  $('network-svg').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); networkClick(e); } });
  document.querySelectorAll('[data-image-view]').forEach(b => b.addEventListener('click', () => { state.imageView = b.dataset.imageView; document.querySelectorAll('[data-image-view]').forEach(x => { x.classList.toggle('active', x === b); x.setAttribute('aria-pressed', x === b); }); render(); }));
  $('specimen-canvas').addEventListener('pointermove', showPatch); $('specimen-canvas').addEventListener('click', showPatch);
  $('patch-index').addEventListener('change', () => { $('patch-index').value = Math.max(1, Math.min(64, Math.floor(+$('patch-index').value) || 1)); readPatch(+$('patch-index').value - 1); });
  $('sample-gallery').addEventListener('click', e => { const b = e.target.closest('[data-sample]'); if (b) selectSample(b.dataset.sample); });
  $('data-gallery').addEventListener('click', e => {
    const b = e.target.closest('[data-sample]'); if (!b) return;
    if (!prepared.some(s => s.id === b.dataset.sample)) { toast('This nucleus is outside the active training subset. Increase the training count to include it.'); return; }
    setStage('train'); selectSample(b.dataset.sample);
  });
  ['shift', 'stain'].forEach(id => $(id).addEventListener('input', () => { pause(); $('follow').checked = false; state.shift = +$('shift').value; state.stain = +$('stain').value; state.occlusion = null; syncStress(); render(); }));
  $('restore-image').addEventListener('click', () => { state.shift = 0; state.stain = 1; state.occlusion = null; syncStress(); render(); });
  document.querySelectorAll('.preset').forEach(b => b.addEventListener('click', () => {
    const name = b.dataset.preset;
    $('input-mode').value = name === 'simple' ? 'shape' : 'pixels'; $('hidden-units').value = name === 'simple' ? 0 : name === 'small' ? 12 : 8;
    $('train-count').value = name === 'small' ? 10 : 80; $('learning-rate').value = name === 'simple' ? 3 : 2;
    state.rate = rates[+$('learning-rate').value]; $('rate-value').textContent = state.rate;
    $('batch-size').value = 8; state.batch = 8; $('l2').value = name === 'small' ? 0 : 0.001; state.l2 = +$('l2').value; $('seed').value = 7;
    clearPreset(); b.classList.add('active'); reset();
  }));
  $('lecture-toggle').addEventListener('click', () => { const active = document.body.classList.toggle('lecture-view'); $('lecture-toggle').setAttribute('aria-pressed', active); $('lecture-toggle').textContent = active ? 'Exit lecture view' : 'Lecture view'; window.scrollTo({ top: 0 }); });
  $('guide-open').addEventListener('click', () => { pause(); $('guide').showModal(); document.body.classList.add('modal-open'); });
  $('guide-close').addEventListener('click', () => $('guide').close()); $('guide').addEventListener('close', () => document.body.classList.remove('modal-open'));
  $('guide').addEventListener('click', e => { const r = $('guide').getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) $('guide').close(); });
  $('export').addEventListener('click', () => {
    pause();
    const report = { format: 'nucleus-lab-experiment-v1', datasetSeed: data.seed, datasetVersion: data.version, settings: { inputMode: state.mode, hidden: state.hidden, weightSeed: state.seed, trainingCount: state.count, learningRate: state.rate, batchSize: state.batch, l2: state.l2, threshold: threshold() }, progress: { epochs: state.epoch, cursor: state.cursor, updates: state.updates }, scaler, weights: model.snapshot(), history, training: trainPredictions.map(({ id, y, p, loss }) => ({ id, label: y, probabilityIrregular: p, loss })), test: state.testResults, testRunsThisSession: state.testRuns, notes: 'Synthetic educational data. Not a clinical model. Current hyperparameters are recorded; mid-run hyperparameter changes are not logged. Repeated test-driven tuning requires a fresh final evaluation set.' };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }), url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `nucleus-lab-${state.mode}-epoch-${state.epoch}.json`; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); toast('Experiment exported, including weights and predictions.');
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden && state.running) { pause(); render(); } });
  $('data-gallery').innerHTML = trainBank.map(s => tile(s, null, true)).join('');
  reset(false);
})();
