/* Pure, dependency-free numerical core. Shared by the browser and Node tests. */
(function (root) {
  'use strict';
  const SIDE = 16;
  const FEATURES = [
    { name: 'Edge complexity', short: 'Edge', detail: 'Perimeter² / (4π × area). Measured on a square pixel grid, so even a smooth circle is above 1.' },
    { name: 'Concavity', short: 'Dent', detail: '1 − area / convex-hull area. Indentations leave empty space inside the smallest enclosing convex outline.' },
    { name: 'Elongation', short: 'Long', detail: 'Ratio of major to minor axes from the foreground pixel moments. A long nucleus is not necessarily irregular.' },
    { name: 'Area', short: 'Size', detail: 'Fraction of pixels occupied by the nucleus. Size alone does not define contour irregularity.' }
  ];
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffle(items, random) {
    const out = [...items];
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }
  function hull(points) {
    const p = points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [], upper = [];
    for (const pt of p) { while (lower.length > 1 && cross(lower.at(-2), lower.at(-1), pt) <= 0) lower.pop(); lower.push(pt); }
    for (let i = p.length - 1; i >= 0; i--) { const pt = p[i]; while (upper.length > 1 && cross(upper.at(-2), upper.at(-1), pt) <= 0) upper.pop(); upper.push(pt); }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }
  function measure(pixels) {
    const mask = pixels.map(v => v >= 0.28 ? 1 : 0);
    let area = 0, perimeter = 0, sx = 0, sy = 0;
    const corners = [];
    for (let y = 0; y < SIDE; y++) for (let x = 0; x < SIDE; x++) {
      if (!mask[y * SIDE + x]) continue;
      area++; sx += x + 0.5; sy += y + 0.5;
      for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        if (x + dx < 0 || x + dx >= SIDE || y + dy < 0 || y + dy >= SIDE || !mask[(y + dy) * SIDE + x + dx]) perimeter++;
      }
      corners.push([x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]);
    }
    if (!area) return [0, 0, 1, 0];
    const h = hull(corners);
    let hullArea = 0, xx = 0, yy = 0, xy = 0;
    for (let i = 0; i < h.length; i++) { const j = (i + 1) % h.length; hullArea += h[i][0] * h[j][1] - h[j][0] * h[i][1]; }
    hullArea = Math.abs(hullArea) / 2;
    for (let y = 0; y < SIDE; y++) for (let x = 0; x < SIDE; x++) if (mask[y * SIDE + x]) {
      const dx = x + 0.5 - sx / area, dy = y + 0.5 - sy / area;
      xx += dx * dx / area; yy += dy * dy / area; xy += dx * dy / area;
    }
    const d = Math.sqrt((xx - yy) ** 2 + 4 * xy * xy);
    return [perimeter ** 2 / (4 * Math.PI * area), Math.max(0, 1 - area / Math.max(area, hullArea)), Math.sqrt((xx + yy + d + 1e-6) / (xx + yy - d + 1e-6)), area / (SIDE * SIDE)];
  }
  const raw = (pixels, mode) => mode === 'shape' ? measure(pixels) : Array.from(pixels);
  function fitScaler(samples, mode) {
    if (!samples.length || samples.some(s => s.split !== 'train')) throw new Error('The scaler accepts training samples only.');
    const xs = samples.map(s => raw(s.pixels, mode)), n = xs[0].length;
    const mean = Array(n).fill(0), scale = Array(n).fill(0);
    for (const x of xs) for (let j = 0; j < n; j++) mean[j] += x[j] / xs.length;
    for (const x of xs) for (let j = 0; j < n; j++) scale[j] += (x[j] - mean[j]) ** 2 / xs.length;
    for (let j = 0; j < n; j++) scale[j] = Math.max(mode === 'pixels' ? 0.15 : 0.01, Math.sqrt(scale[j]));
    return { mean, scale, fittedIds: samples.map(s => s.id) };
  }
  function encode(pixels, mode, scaler) { return raw(pixels, mode).map((v, i) => (v - scaler.mean[i]) / scaler.scale[i]); }
  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  const lossFromLogit = (z, y) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
  class Network {
    constructor(inputs, hidden = 0, seed = 7) {
      this.inputs = inputs; this.hidden = hidden;
      const random = rng(seed), units = hidden || 1;
      this.w1 = Array.from({ length: inputs * units }, () => (random() * 2 - 1) * Math.sqrt(6 / (inputs + units)));
      this.b1 = Array(units).fill(0);
      this.w2 = hidden ? Array.from({ length: hidden }, () => (random() * 2 - 1) * Math.sqrt(6 / (hidden + 1))) : [];
      this.b2 = [0];
    }
    forward(x) {
      const z1 = [], a = [];
      for (let j = 0; j < (this.hidden || 1); j++) {
        let z = this.b1[j];
        for (let i = 0; i < this.inputs; i++) z += this.w1[j * this.inputs + i] * x[i];
        z1.push(z); a.push(this.hidden ? Math.tanh(z) : sigmoid(z));
      }
      let z = z1[0];
      if (this.hidden) { z = this.b2[0]; for (let j = 0; j < this.hidden; j++) z += this.w2[j] * a[j]; }
      return { x, z1, a, z, p: sigmoid(z) };
    }
    gradients(batch, l2 = 0) {
      if (!batch.length) throw new Error('Empty batch');
      const g = { w1: this.w1.map(() => 0), b1: this.b1.map(() => 0), w2: this.w2.map(() => 0), b2: [0] };
      let loss = 0;
      for (const s of batch) {
        const f = this.forward(s.x), dz = (f.p - s.y) / batch.length;
        loss += lossFromLogit(f.z, s.y) / batch.length;
        if (this.hidden) {
          g.b2[0] += dz;
          for (let j = 0; j < this.hidden; j++) {
            g.w2[j] += dz * f.a[j];
            const dh = dz * this.w2[j] * (1 - f.a[j] ** 2);
            g.b1[j] += dh;
            for (let i = 0; i < this.inputs; i++) g.w1[j * this.inputs + i] += dh * s.x[i];
          }
        } else {
          g.b1[0] += dz;
          for (let i = 0; i < this.inputs; i++) g.w1[i] += dz * s.x[i];
        }
      }
      for (const key of ['w1', 'w2']) for (let i = 0; i < this[key].length; i++) { g[key][i] += l2 * this[key][i]; loss += l2 * this[key][i] ** 2 / 2; }
      return { g, loss };
    }
    trainBatch(batch, rate, l2 = 0) {
      if (batch.some(s => s.split !== 'train')) throw new Error('Test samples cannot enter backpropagation.');
      const { g, loss } = this.gradients(batch, l2);
      for (const key of ['w1', 'b1', 'w2', 'b2']) for (let i = 0; i < this[key].length; i++) this[key][i] -= rate * g[key][i];
      if (this.parameters().some(v => !Number.isFinite(v))) throw new Error('Weights diverged. Reset and reduce the learning rate.');
      return { g, loss };
    }
    parameters() { return [...this.w1, ...this.b1, ...this.w2, ...(this.hidden ? this.b2 : [])]; }
    snapshot() { return { inputs: this.inputs, hidden: this.hidden, w1: [...this.w1], b1: [...this.b1], w2: [...this.w2], b2: [...this.b2] }; }
  }
  function metrics(predictions, threshold = 0.5) {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const s of predictions) { const p = s.p >= threshold; if (s.y) { if (p) tp++; else fn++; } else { if (p) fp++; else tn++; } }
    return { tp, tn, fp, fn, total: predictions.length, accuracy: (tp + tn) / predictions.length, sensitivity: tp + fn ? tp / (tp + fn) : null, specificity: tn + fp ? tn / (tn + fp) : null };
  }
  function occlusion(model, pixels, mode, scaler) {
    const base = model.forward(encode(pixels, mode, scaler)).p, deltas = [];
    for (let py = 0; py < SIDE; py += 2) for (let px = 0; px < SIDE; px += 2) {
      const masked = Array.from(pixels);
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) masked[(py + dy) * SIDE + px + dx] = 0;
      deltas.push(base - model.forward(encode(masked, mode, scaler)).p);
    }
    return { base, deltas };
  }
  function transform(pixels, shift = 0, stain = 1) {
    return pixels.map((_, i) => { const x = i % SIDE - shift, y = Math.floor(i / SIDE); return x < 0 || x >= SIDE ? 0 : Math.max(0, Math.min(1, pixels[y * SIDE + x] * stain)); });
  }
  const api = { SIDE, FEATURES, rng, shuffle, measure, raw, fitScaler, encode, sigmoid, lossFromLogit, Network, metrics, occlusion, transform };
  root.NucleusCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
