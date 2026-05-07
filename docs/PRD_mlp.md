# MLP PRD

## Goal
Classify sine-wave frequencies and reconstruct clean signals using a fully-connected neural network.

## Inputs
- 10-sample noisy signal window.

## Outputs
- Frequency logits.
- Reconstructed clean window.

## Notes
The MLP ignores temporal memory and processes the window as a flat vector.