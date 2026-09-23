const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const C = require('../src/core.js');
require('../data/nuclei.js');
const samples = globalThis.NUCLEI_DATA.samples;
const train = samples.filter(s => s.split === 'train'), heldOut = samples.filter(s => s.split === 'test');

function experiment(mode, hidden, count = 80, epochs = 100) {
  const active = train.slice(0, count), scaler = C.fitScaler(active, mode);
  const prepared = active.map(s => ({ id: s.id, split: s.split, x: C.encode(s.pixels, mode, scaler), y: s.label }));
  const model = new C.Network(mode === 'shape' ? 4 : 256, hidden, 7), random = C.rng(110);
  const loss = () => prepared.reduce((v, s) => v + C.lossFromLogit(model.forward(s.x).z, s.y), 0) / prepared.length;
  const initial = loss();
  for (let epoch = 0; epoch < epochs; epoch++) {
    const order = C.shuffle(prepared, random);
    for (let start = 0; start < order.length; start += 8) model.trainBatch(order.slice(start, start + 8), mode === 'shape' ? 0.08 : 0.03, 0.001);
  }
  const metrics = list => C.metrics(list.map(s => ({ y: s.label, p: model.forward(C.encode(s.pixels, mode, scaler)).p })));
  return { model, scaler, initial, final: loss(), trainMetrics: metrics(active), testMetrics: metrics(heldOut) };
}

test('100 fixed, unique images; balanced 80/20 split, with shape pairs kept together', () => {
  assert.equal(samples.length, 100); assert.equal(train.length, 80); assert.equal(heldOut.length, 20);
  assert.equal(train.filter(s => s.label).length, 40); assert.equal(heldOut.filter(s => s.label).length, 10);
  assert.equal(new Set(samples.map(s => s.id)).size, 100);
  assert.equal(new Set(samples.map(s => JSON.stringify(s.pixels))).size, 100);
  const trainPairs = new Set(train.map(s => s.pair)); assert(heldOut.every(s => !trainPairs.has(s.pair)));
  for (const s of samples) {
    assert.equal(s.pixels.length, 256); assert(s.pixels.every(v => Number.isFinite(v) && v >= 0 && v <= 1));
    assert(fs.existsSync(path.join(__dirname, '..', 'nuclei', s.id + '.svg')));
    assert(C.measure(s.pixels).every(Number.isFinite));
  }
});

test('single neuron and tanh network gradients match finite differences, including L2 and biases', () => {
  for (const hidden of [0, 3]) {
    const model = new C.Network(4, hidden, 17);
    const batch = [{ x: [0.4, -0.3, 0.8, 0.2], y: 1, split: 'train' }, { x: [-0.4, 0.1, 0.4, 0.7], y: 0, split: 'train' }];
    const { g } = model.gradients(batch, 0.013), eps = 1e-5;
    for (const key of ['w1', 'w2', 'b1', ...(hidden ? ['b2'] : [])]) for (let i = 0; i < model[key].length; i++) {
      const old = model[key][i]; model[key][i] = old + eps; const plus = model.gradients(batch, 0.013).loss;
      model[key][i] = old - eps; const minus = model.gradients(batch, 0.013).loss; model[key][i] = old;
      assert(Math.abs((plus - minus) / (2 * eps) - g[key][i]) < 1e-7, `${hidden} ${key}[${i}] gradient mismatch`);
    }
  }
});

test('training-only guards protect both the scaler and backpropagation', () => {
  assert.throws(() => C.fitScaler(heldOut, 'shape'), /training samples only/);
  const scaler = C.fitScaler(train.slice(0, 10), 'shape');
  assert.deepEqual(scaler.fittedIds, train.slice(0, 10).map(s => s.id));
  const model = new C.Network(4);
  const before = model.snapshot();
  assert.throws(() => model.trainBatch([{ x: [0, 0, 0, 0], y: 1, split: 'test' }], 0.1), /Test samples/);
  assert.deepEqual(model.snapshot(), before);
});

test('real training lowers loss in both teaching presets, with deterministic results', () => {
  for (const [mode, hidden] of [['shape', 0], ['pixels', 8]]) {
    const a = experiment(mode, hidden), b = experiment(mode, hidden);
    assert(a.final < a.initial * 0.25, `${mode}: loss did not drop enough (${a.initial} → ${a.final})`);
    assert(a.trainMetrics.accuracy >= 0.95, `${mode}: should fit this synthetic teaching dataset`);
    assert.deepEqual(a.model.snapshot(), b.model.snapshot());
    console.log(`${mode} / hidden ${hidden}: loss ${a.initial.toFixed(4)} → ${a.final.toFixed(4)}, train ${a.trainMetrics.accuracy * 100}%, test ${a.testMetrics.accuracy * 100}%`);
  }
});

test('test evaluation, occlusion and threshold changes leave all parameters untouched', () => {
  const { model, scaler } = experiment('pixels', 8, 80, 5), before = model.snapshot();
  const predictions = heldOut.map(s => ({ y: s.label, p: model.forward(C.encode(s.pixels, 'pixels', scaler)).p }));
  const loose = C.metrics(predictions, 0.1), strict = C.metrics(predictions, 0.9);
  assert(loose.tp + loose.fp >= strict.tp + strict.fp);
  assert.equal(loose.tp + loose.tn + loose.fp + loose.fn, 20);
  const map = C.occlusion(model, heldOut[0].pixels, 'pixels', scaler);
  assert.equal(map.deltas.length, 64); assert(map.deltas.every(Number.isFinite));
  const original = heldOut[0].pixels, masked = [...original];
  for (const i of [0, 1, 16, 17]) masked[i] = 0;
  assert.equal(map.deltas[0], map.base - model.forward(C.encode(masked, 'pixels', scaler)).p);
  assert.deepEqual(model.snapshot(), before); assert.deepEqual(heldOut[0].pixels, original);
});

test('shape-mode occlusion recomputes measurements, and transforms do not mutate the dataset', () => {
  const { model, scaler } = experiment('shape', 0, 80, 5), sample = train[1], before = [...sample.pixels];
  const map = C.occlusion(model, sample.pixels, 'shape', scaler), k = 27, masked = [...sample.pixels];
  const x = (k % 8) * 2, y = Math.floor(k / 8) * 2;
  for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) masked[(y + dy) * 16 + x + dx] = 0;
  assert.equal(map.deltas[k], map.base - model.forward(C.encode(masked, 'shape', scaler)).p);
  C.transform(sample.pixels, 3, 1.4); assert.deepEqual(sample.pixels, before);
  assert(C.measure(Array(256).fill(0)).every(Number.isFinite));
  assert.equal(C.sigmoid(1000), 1); assert.equal(C.sigmoid(-1000), 0);
  assert(Number.isFinite(C.lossFromLogit(-1000, 1)));
});
