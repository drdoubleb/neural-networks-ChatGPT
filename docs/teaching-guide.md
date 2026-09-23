# Presenter guide: how a neuron learns nuclear irregularity

Allow 10–15 minutes. This exercise introduces supervised learning before convolutional networks or transformers. **Lecture view** in the header hides the introductory banner and enlarges the model for projection.

1. Open **Meet the nuclei**. Ask residents to point out a smooth and an irregular contour. State the exact task: contour shape, not malignancy. Labels were assigned by the synthetic shape generator.
2. Return to **Train a network → One neuron**. Four measurements enter: edge complexity, concavity, elongation and area. We supplied these useful measurements; the neuron did not discover them. A simple model can perform well when someone has already designed informative inputs.
3. Inspect a nucleus. Values at the input nodes are standardized using the active training subset. Each value is multiplied by a weight. Sum the products, add a bias and apply a sigmoid to obtain P(irregular). Open **Show the calculation** and inspect the actual arithmetic.
4. Click **1 update**. Training uses the known label (0 regular, 1 irregular). The panel gives the first example's pre-update score, its error, and one weight's actual before/after values. The gradient is averaged over the batch and includes L2. The images and node activations above use the updated weights.
5. Click **Start training**, and watch the weight signs, their magnitudes and the training loss. A loss decrease is more informative than accuracy alone; a probability can improve without crossing the threshold. One epoch means every active training image has been used once. Use Fast for a brisk lecture.
6. Pause. Select a misclassified nucleus or filter to **Mistakes only**. Inspect the score and threshold. A threshold change is a decision-rule change, not additional learning.
7. Try **Learn from pixels**. This resets the model. The inputs are now the 256 raster intensities, with no explicit edge or concavity measurements. Eight tanh neurons form the hidden layer. Each hidden neuron makes a nonlinear combination of all inputs; the output neuron combines their activations. Select hidden neurons to inspect their different weight maps. Do not name a unit an “indentation detector” without evidence.
8. Switch the input image to **Influence**. This is an occlusion experiment: remove each 2 × 2 patch and measure the change in P(irregular). Hover or tap patches for numerical effects. Discuss why erasing pixels can create an unnatural image, especially when recomputing shape measurements. A colorful overlay is not proof that the model is using a meaningful biological feature.
9. Open **Test what it learned**, then run the fixed model on the 20 reserved nuclei. There is no backpropagation or scaler fitting on these images. Read the confusion matrix: reference labels are rows, predictions are columns. A false negative here is an irregular nucleus called regular, not a missed cancer.
10. Move the threshold. Ask what happens to irregular sensitivity and regular specificity. Probability outputs and weights are unchanged. Each test nucleus changes accuracy by 5 percentage points, so this is a coarse estimate.
11. Optional: try **Less data**, a large learning rate, or a different weight seed. The ten-image subset is balanced. Discuss training accuracy versus generalization. These experiments are illustrative; do not promise that a particular seed or model will always win. Trying many settings against this same test set contaminates its role as a final evaluation set.
12. Optional: shift a selected nucleus or change its stain. The original reference label is unchanged. The displayed score changes, but these modified images do not enter the aggregate metrics. An image-based model can be sensitive to position because it has no translation-invariance mechanism. Strong stain changes can also alter the thresholded mask used by shape mode.

## The small amount of math worth showing

For the single neuron:

`z = Σ(wᵢ × xᵢ) + b`

`p = sigmoid(z) = 1 / (1 + exp(−z))`

`loss = −y log(p) − (1−y) log(1−p)`

`∂loss/∂wᵢ = (p − y) × xᵢ` for one example, before L2.

`new weight = old weight − learning rate × gradient`

For a hidden unit: `aⱼ = tanh(Σ(wⱼᵢ × xᵢ) + bⱼ)`. Its derivative is `1 − aⱼ²`. The output remains a sigmoid; backpropagation uses the chain rule. Removing all nonlinear hidden activations would collapse the composition into another affine function, regardless of depth.

## Where to go next

- **Convolution:** reuse a local filter across locations instead of a separate weight for every fixed pixel position.
- **Pooling and augmentation:** discuss mechanisms that can reduce sensitivity to translation and acquisition changes.
- **Representation learning:** contrast supplied shape measurements with learned image features.
- **Real pathology evaluation:** split by patient, consider site and scanner differences, label quality, class balance, clinical consequences and independent validation.
- **Attention and transformers:** introduce them as different architectures; the occlusion overlay in this lab is not attention.

## Practical notes

The bank is fixed at 100 images. The 80/20 split is pre-created, with paired nuclei kept together. The training-count control selects a balanced prefix of the training bank and refits the scaler on that subset only. Controls that change the architecture, input representation, seed or subset reset the model and clear test results. Learning rate, L2 and batch size can change during training. A new weight update clears any previous test evaluation to avoid displaying stale results. The browser session is ephemeral; use **Export experiment JSON** to retain the model and results. A normal refresh resets the lab.

The app can be projected from `index.html` without an internet connection. No audio is required; discuss the changes while residents operate the controls.
