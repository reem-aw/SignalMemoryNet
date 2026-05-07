# Product Requirements Document

## Project Goal
Build a neural-network system that receives noisy sine-wave windows and performs:
1. Frequency classification.
2. Clean signal reconstruction.

## Inputs
- Noisy sine-wave windows of length 10.
- Frequencies: 2, 5, 10, 20 Hz.
- Sampling rate: 100 Hz.
- Noise levels: 5%, 10%, 20%.

## Outputs
- One-hot frequency prediction (4 classes).
- Reconstructed clean signal window.

## Architectures
- MLP
- Vanilla RNN
- LSTM

## Evaluation
- Classification accuracy
- Reconstruction MSE
- Confusion matrices
- Reconstruction plots

## Functional Requirements
- Generate synthetic sine-wave signals.
- Add configurable Gaussian noise.
- Classify the signal frequency.
- Reconstruct the clean signal.
- Train and evaluate three neural-network architectures.

## Acceptance Criteria
- All models train successfully.
- Test accuracy exceeds 95%.
- Reconstruction MSE is low and stable.
- Unit tests pass successfully.