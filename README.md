# SignalMemoryNet — HW1 Lab Report

Predicting and reconstructing noisy sine signals with three neural-network
architectures: a fully-connected **MLP** (no memory), a vanilla **RNN**
(implicit memory via hidden state), and an **LSTM** (gated long/short-term
memory).

> **Repository:** <https://github.com/reem-aw/SignalMemoryNet>
>
> **Course:** Neural Networks — HW1 (sine recognition / reconstruction)

---

## 1. Problem statement

We model time-series prediction with the recurrence

$$
y_{t+1} \;=\; f_{\mathbf{w}}\bigl(\mathbf{x}_{t},\;\mathbf{m}_{t}\bigr)
$$

where $\mathbf{m}_t$ is the previous state ("memory").  In the homework we
restrict the memory window to a fixed **sliding context window** of $W=10$
samples, the analogue of a context window in modern LLMs.  Each input is a
length-10 *noisy* slice of a sine wave; the network must

1. **classify** which of the four known frequencies generated the slice
   (1-hot output of length 4), and
2. **reconstruct** the corresponding *clean* slice (length 10).

Both objectives are trained jointly with a sum of cross-entropy and MSE losses.

## 2. Theory recap

### 2.1 Sampling
By the Nyquist–Shannon theorem we must sample at $f_s \ge 2 f_{\max}$. We pick
$f_s = 100$ Hz and frequencies $\{2, 5, 10, 20\}$ Hz so the highest frequency is
well below Nyquist (50 Hz), guaranteeing reconstructibility.

### 2.2 Signal model
Following the HW spec we generate

$$
s(t)=A\sin(2\pi f t+\varphi)+\eta(t),
\qquad
\eta(t)\sim\mathcal{N}(0,(\sigma\cdot A)^2)
$$

with $A\in[0.8,1.2]$, $\varphi\in[0,2\pi)$ and $\sigma\in\{5\%,10\%,20\%\}$.
Each record is generated **twice** with shared $A,\varphi$: once clean
(target) and once noisy (input).

### 2.3 Memory layer / RNN
A vanilla RNN augments each layer with a "memory layer" — at every step the
input is concatenated with the previous hidden activations.  For an input
$\mathbf{x}\in\mathbb{R}^n$ and a layer with $p$ perceptrons the effective
input dimension becomes $n+p$.  This is exactly what `torch.nn.RNN`
implements: $h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$.

### 2.4 Vanishing gradient & LSTM
RNNs propagate gradients through repeated multiplication of $W_{hh}$, causing
gradients to vanish (or explode) for long sequences.  This makes vanilla RNNs
poor at long-range memory but *adequate for short, high-frequency signals* —
fewer samples per period need to be retained.  The **LSTM** mitigates the
vanishing-gradient problem with three multiplicative ("element-wise") gates
(input, forget, output) over a separate cell-state $c_t$, allowing each
component of the hidden vector to be selectively opened/closed and giving the
network the capacity to capture different frequency components in its
representation.

## 3. Dataset

| Parameter | Value |
|---|---|
| Frequencies | 2, 5, 10, 20 Hz |
| Sampling rate $f_s$ | 100 Hz |
| Record length | 10 s = 1000 samples |
| Context window | 10 samples |
| Window stride | 5 (training) |
| Noise levels $\sigma$ | 5 %, 10 %, 20 % of $A$ |
| Amplitude range | 0.8 – 1.2 |
| Phase | uniform $[0, 2\pi)$ |

Each item supplied to the network is a dict `{x_noisy, x_clean, y_class,
y_onehot, sigma}`.  Train / val / test splits use 30 / 8 / 8 records per
frequency by default (see [`src/config.py`](config/config.py)).

## 4. Architectures

All three models share the same multi-task head — a single linear layer that
projects the backbone features to 14 outputs, sliced into 4 logits and 10
reconstruction values (see [`src/models/heads.py`](src/models/heads.py)).

| Model | Backbone | Params input |
|---|---|---|
| MLP  | `Linear(10, 64) → ReLU → Linear(64, 64) → ReLU` | flattened 10-vector |
| RNN  | `RNN(in=1, hidden=32, tanh)` over 10 timesteps | sample-as-timestep |
| LSTM | `LSTM(in=1, hidden=32)` over 10 timesteps | sample-as-timestep |

The RNN/LSTM treat each of the 10 samples as one timestep with feature
dimension 1 — directly mirroring the lecture's "10-sample context window".

## 5. Training

* Optimiser: **Adam**, lr = 1e-3
* Loss: $\mathcal{L} = \lambda_{cls}\,\text{CE}(\hat y, y) + \lambda_{rec}\,\text{MSE}(\hat x_{clean}, x_{clean})$ with $\lambda_{cls}=\lambda_{rec}=1$.
* Batch size 128, up to 20 epochs, early stopping on validation loss
  (patience = 5).
* Best-checkpoint selection on validation loss.

Run `python -m src.main --model all --epochs 30` to retrain every model.

## 6. Results

After running the training script, `report/figures/` contains:

* `training_curves.png` — train/val loss, accuracy and recon-MSE per epoch
  for all three models.
* `confusion_<model>.png` — per-frequency confusion matrices on the test set.
* `recon_<model>.png` — sample reconstructions overlayed on the clean target.

The numerical summary is written to `report/metrics.json`.

### Expected qualitative findings

* The MLP can already classify frequencies well from a 10-sample window
  (the four frequencies are far apart in spectral terms) but its
  reconstruction is the noisiest.
* The vanilla RNN matches or exceeds the MLP on classification and produces
  smoother reconstructions, especially for the **high-frequency (20 Hz)**
  signal — where less long-range memory is needed (lecture remark on
  short-term memory suiting high-frequency content).
* The LSTM is the most stable across noise levels and performs best on the
  lower frequencies, where more samples per period must be retained to
  disambiguate the underlying signal.

## 7. How to run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Tests (~150 lines across three files)
pytest

# Train + evaluate all three models and emit figures + metrics
python -m src.main --model all --epochs 20

```

## 8. Repository layout

```
src/
  config.py          # frequencies, fs, window, hyperparameters
  data.py            # generate_signal + SignalWindowDataset
  train.py           # shared training loop with early stopping
  evaluate.py        # plotting + metric helpers
  main.py            # CLI entry point
  models/
    heads.py         # MultiTaskHead (4 logits + 10 reconstruction)
    mlp.py
    rnn.py
    lstm.py
tests/               # pytest suite (~150 lines)
report/
  build_pdf.py       # generates Submission.pdf
  figures/           # generated by training
```

## 9. Submission

Exercise **1**

Group: **SMNGRP05**

| Student | ID | Name (EN) | Name (HE) |
|---|---|---|---|
| 1 | 208123232 | Afaf Gharra  | עפאף גרה |
| 2 | 212018899 | Reem Awawdy  | רים עואודה |

GitHub: <https://github.com/reem-aw/SignalMemoryNet>
Late submission: **No**.
