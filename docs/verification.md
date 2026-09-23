# Verification

Verified on 23 September 2026 with Node.js 24 and Chromium 134.

## Numerical checks

`npm test` passes all six tests. The checks cover the fixed 100-image bank, balanced and disjoint splits, pair grouping, finite-difference agreement for both linear and hidden-layer gradients, L2 and bias gradients, deterministic training, training-only scaler fitting, rejection of test samples by the training method, exact patch-removal effects, and unchanged weights during evaluation.

With weight seed 7, batch size 8 and 100 epochs:

| Model | Learning rate | L2 | Initial loss | Final loss | Train accuracy | Test accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Four measurements → one sigmoid neuron | 0.08 | 0.001 | 1.8725 | 0.1069 | 95% | 18/20 |
| 256 pixels → 8 tanh units → one sigmoid neuron | 0.03 | 0.001 | 0.8056 | 0.0150 | 100% | 18/20 |

These are reproducible toy-dataset results, not a performance guarantee or a clinical validation. Test accuracy was not used as a test-suite success threshold.

## Browser checks

The exercised flow included:

- Initial load and all image assets, with no browser JavaScript errors or failed HTTP responses.
- One update, one epoch, continuous training, pause/resume, and exact automatic pause at epoch 200 in both default presets.
- All 80 training predictions, 20 test predictions, threshold-dependent confusion counts, specimen selection, hidden-neuron selection, and individual pixel weight inspection.
- Exported weights compared before and after held-out evaluation and threshold changes: identical.
- Patch-removal display, numerical patch selection and updates after changing the selected specimen.
- Temporary image shift and restoration, without changing the underlying dataset or aggregate metrics.
- Clearing stale test results after new training, followed by restoration of all training thumbnails.
- Reduced training sets and a requested batch size larger than the active set.
- The teaching dialog, Escape to close it, experiment JSON contents and an actual completed JSON download.
- Lecture layout, 820-pixel tablet layout, 390-pixel phone layout without page-wide overflow, and horizontal scrolling of the phone network diagram.
- Direct offline use through `file://`, including training and download.

The 200-epoch browser runs reached loss 0.0940 for the measurement model and 0.0069 for the pixel model. Desktop and phone renderings were visually inspected. The preview image shows a separate 30-epoch pixel-model run.

Browser automation was used during implementation; the committed test suite is the dependency-free numerical suite. Safari and Firefox were not separately exercised.
