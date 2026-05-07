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