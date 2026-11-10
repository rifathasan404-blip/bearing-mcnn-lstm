# Chen et al. (2021) — Reproduction Spec Sheet

**Paper:** Chen, X., Zhang, B., Gao, D. (2021). *Bearing fault diagnosis base on multi-scale CNN and LSTM model.* Journal of Intelligent Manufacturing 32, 971–987. doi:10.1007/s10845-020-01600-2

**Page column:** PDF page (journal page = PDF page + 970). "Fig./Table/Eq." points to the exact spot.

**Status key**
- **STATED**: the paper says it explicitly.
- **INFERRED**: not written, but follows from the paper's own numbers (the reasoning is shown).
- **UNKNOWN**: not recoverable from the paper. Each one is a reproduction risk (see section 7).

---

## 1. Data

| # | Parameter | Value | Page | Status |
|---|---|---|---|---|
| D1 | Dataset | Case Western Reserve University (CWRU) bearing data | 5–6, 8 | STATED |
| D2 | Bearing | SKF 6205-2RS JEM deep-groove ball bearing | 5 | STATED |
| D3 | Sensor / channel | Accelerometer at the drive end (DE) | 8; Table 1 (p.7) | STATED |
| D4 | Variable in the `.mat` file | `X###_DE_time` (number zero-padded to 3 digits, e.g. `X097_DE_time`) | — | INFERRED (follows from D3); naming confirmed in our files, see 1a |
| D5 | Sampling rate | 48 kHz | 8 | STATED |
| D6 | Motor load / speed | 3 hp, 1730 rpm | 8; Table 1 | STATED |
| D7 | Fault types | Ball (rolling element), inner race, outer race | 8 | STATED |
| D8 | Defect diameters | 0.007, 0.014, 0.021 in | 8; Table 1 | STATED |
| D9 | Outer-race fault position | 6:00 (load zone) | Table 1 | STATED |
| D10 | Number of classes | 10 = 3 types × 3 sizes + 1 healthy | 5, 8 | STATED |
| D11 | Label mapping | 0 B007 · 1 B014 · 2 B021 · **3 Normal** · 4 IR007 · 5 IR014 · 6 IR021 · 7 OR007@6 · 8 OR014@6 · 9 OR021@6 | Table 1 | STATED |
| D12 | CWRU file numbers | 125, 192, 229, 100, 112, 177, 217, 138, 204, 241 (same order as D11) | — | INFERRED (from D5–D11 + our manifest) |
| D13 | Sampling rate of normal file `100.mat` | Paper implies 48 kHz for all files; CWRU site does not state it | 8 | UNKNOWN (verify by sample count) |

### 1a. Measured from our own files (not from the paper)

Source: `notebooks/00_data_inventory.ipynb`, run 2026-11-10. These rows are **MEASURED**: facts read from the `.mat` files, independent of what Chen states. Neither file is one of Chen's ten (D12); they were opened to learn the file layout.

| # | What was checked | `97.mat` (Normal, 0 hp) | `105.mat` (IR007, 0 hp, 12k drive end) |
|---|---|---|---|
| M1 | Folder | `data/raw/normal` | `data/raw/12k_drive_end` |
| M2 | Signal keys | `X097_DE_time`, `X097_FE_time` | `X105_DE_time`, `X105_FE_time`, `X105_BA_time` |
| M3 | Base (BA) channel present? | **No** | Yes |
| M4 | Speed key | `X097RPM`, shape (1, 1), uint16 | `X105RPM`, shape (1, 1), uint16 |
| M5 | Signal shape / dtype | (243938, 1), float64 | (121265, 1), float64 |
| M6 | Readings in `DE_time` | 243 938 | 121 265 |
| M7 | Sampling rate used for M8 | 48 000 Hz (**assumed**, typed by hand) | 12 000 Hz (from the folder name) |
| M8 | Duration = M6 / M7 | 5.08 s | 10.11 s |

What this means for the loader and the spec:

- **Signals are column vectors** of shape (N, 1). The loader must flatten them to (N,).
- **The key set differs between files** (no `BA_time` in `97.mat`). The loader must find the DE key by its `DE_time` suffix and must not assume a fixed list of keys.
- **The RPM value itself was not printed**, only its shape. Print it for each file and compare with D6.
- **D13 is still open.** A sample count cannot prove a sampling rate: 243 938 readings is 5.08 s at 48 kHz or 20.33 s at 12 kHz, and both are possible. To settle it, find the shaft-frequency peak in the FFT of `100.mat` and check it sits at RPM / 60.
- **W5 needs the lengths of Chen's ten files.** 240 non-overlapping windows need 480 000 readings. `97.mat` has only 243 938, so run the same count on the ten files in D12 before deciding whether non-overlap is possible.

## 2. Sample construction (windowing)

| # | Parameter | Value | Page | Status |
|---|---|---|---|---|
| W1 | Window length (before down-sampling) | 2000 consecutive points | 8; Table 1 | STATED |
| W2 | Duration of one window | 2000 / 48 000 ≈ 41.7 ms ≈ 1.2 shaft revolutions at 1730 rpm | — | INFERRED |
| W3 | Samples per class | 240 | 8; Table 1 | STATED |
| W4 | Total samples | 2400 | 8 | STATED (1680 + 720) |
| W5 | Window stride / overlap | Not given. 240 × 2000 = 480 000 points, which fits in one ~10 s 48 kHz recording **without overlap** | — | UNKNOWN (non-overlap is plausible; check the file lengths) |
| W6 | Window start (random vs. sequential) | Not given | — | UNKNOWN |
| W7 | Train / test split | 70 % / 30 % → 1680 train, 720 test | 8 | STATED |
| W8 | Split unit | Random split of **windows**, so train and test windows come from the same recording | 8 | INFERRED |
| W9 | Stratified per class? | **No.** Test counts per class in Fig. 8a are 71, 78, 66, 75, 64, 64, 76, 78, 68, 80 (sum 720), not 72 each | Fig. 8 (p.9) | INFERRED |
| W10 | Split re-drawn each trial? | Probably. Fig. 8a and 8b have different per-class counts | Fig. 8 | INFERRED |
| W11 | Validation set | None mentioned | — | UNKNOWN |

## 3. Preprocessing

| # | Parameter | Value | Page | Status |
|---|---|---|---|---|
| P1 | Down-sampling | Keep 1 of every 8 points → 250 points per sample | 8; Fig. 6 (p.6) | STATED |
| P2 | Effective sampling rate / Nyquist | 48 000 / 8 = 6 kHz → content above 3 kHz is lost or aliased | — | INFERRED |
| P3 | Anti-alias filter before decimation | Not mentioned ("selecting one datapoint of 8" reads like plain slicing `x[::8]`) | 8 | UNKNOWN |
| P4 | Decimation offset (start at index 0?) | Not given | — | UNKNOWN |
| P5 | Normalisation / scaling | Not mentioned. Fig. 6 amplitudes stay in ±2, the same as raw | 6 | UNKNOWN |
| P6 | Feature engineering | None, raw signal in ("without preprocessing") | 1, 2, 16 | STATED |
| P7 | Data augmentation | None mentioned | — | UNKNOWN |
| P8 | Network input shape | 1 × 250 | Table 2 (p.8) | STATED |

## 4. Architecture (Table 2, p.8; Fig. 4, p.5)

All convolutions are 1-D. The output lengths match `(L − k)/s + 1` with **no padding** ("valid"), which is INFERRED from the numbers. Every row's parameter count was recomputed and matches Table 2.

### Branch CNN_1: large kernels, "low-frequency" features

| # | Layer | Filters | Kernel / stride | Activation | Output | Params | Status |
|---|---|---|---|---|---|---|---|
| A1 | Conv_1 | 50 | 20 / 2 | tanh | 116 × 50 | 1 050 | STATED |
| A2 | Conv_2 | 30 | 10 / 2 | tanh | 54 × 30 | 15 030 | STATED |
| A3 | Pooling_1 (max) | — | 2 / 2 | — | 27 × 30 | 0 | STATED |

### Branch CNN_2: small kernels, deeper, "high-frequency" features

| # | Layer | Filters | Kernel / stride | Activation | Output | Params | Status |
|---|---|---|---|---|---|---|---|
| A4 | Conv_3 | 50 | 6 / 1 | tanh | 245 × 50 | 350 | STATED (see challenge 1 for the input) |
| A5 | Conv_4 | 40 | 6 / 1 | tanh | 240 × 40 | 12 040 | STATED |
| A6 | Pooling_2 (max) | — | 2 / 2 | — | 120 × 40 | 0 | STATED |
| A7 | Conv_5 | 30 | 6 / 1 | tanh | 115 × 30 | 7 230 | STATED |
| A8 | Conv_6 | 30 | 6 / 2 | tanh | 55 × 30 | 5 430 | STATED |
| A9 | Pooling_3 (max) | — | 2 / 2 | — | 27 × 30 | 0 | STATED (55/2 floored) |

### Fusion and classifier

| # | Parameter | Value | Page | Status |
|---|---|---|---|---|
| A10 | Branch input | Both branches get the **same** 1 × 250 signal in parallel | 4 | STATED |
| A11 | Feature fusion | **Element-wise product** of the two 30 × 27 maps | 4 | STATED |
| A12 | Pooling type | Max pooling | 4 (Eq. 6) | STATED |
| A13 | LSTM_1 | 60 units, returns the full sequence → 30 × 60, activation ReLU, 21 120 params | Table 2 | STATED |
| A14 | LSTM_1 sequence axis | 30 time steps × 27 features, so the LSTM steps over the **30 filter channels**, not over time (see puzzle 3) | — | INFERRED (from the param count) |
| A15 | LSTM recurrent activation | Not given. The Keras 2.2.4 default was `hard_sigmoid` | — | UNKNOWN |
| A16 | LSTM_2 | 30 units, last state only → 1 × 30, ReLU, 10 920 params | Table 2 | STATED |
| A17 | Dropout | After LSTM_2 | 8 | STATED |
| A18 | Dropout rate | Not given | — | UNKNOWN |
| A19 | Dense output | 10 units, softmax, 310 params | Table 2; Eq. 12 | STATED |
| A20 | Total trainable params | 73 480 (recomputed: matches) | Table 2 | STATED |
| A21 | Weight initialisation | Not given (paper only says init matters) | 8 | UNKNOWN |
| A22 | Batch norm / regularisers | None mentioned | — | UNKNOWN |

## 5. Training

| # | Parameter | Value | Page | Status |
|---|---|---|---|---|
| T1 | Framework | Keras 2.2.4, TensorFlow backend | 8 | STATED |
| T2 | Hardware | Intel i7-8565U CPU, 8 GB RAM (no GPU) | 8 | STATED |
| T3 | Loss | **Mean squared error** (unusual: softmax classifiers normally use cross-entropy) | 8 | STATED |
| T4 | Label encoding | One-hot (needed for MSE against a softmax) | — | INFERRED |
| T5 | Optimizer | Not named ("back propagation to update the weights") | 8 | UNKNOWN |
| T6 | Learning rate | 0.0006 | 8 | STATED |
| T7 | Epochs | 400 | 8 | STATED |
| T8 | Batch size | Not given | — | UNKNOWN |
| T9 | LR schedule / early stopping | Not mentioned | — | UNKNOWN |
| T10 | Independent trials | 10, same hyperparameters | 8–9 | STATED |
| T11 | Random seeds | Not given | — | UNKNOWN |
| T12 | Train accuracy at the end | 100 % in all 10 trials (Fig. 7) | 9 | STATED |

## 6. Evaluation and reported results

| # | Parameter | Value | Page | Status |
|---|---|---|---|---|
| E1 | Metrics | Accuracy, precision, recall, F1 | 8 (Eqs. 13–16) | STATED |
| E2 | Aggregation | Per-class precision/recall/F1, averaged over 10 trials | 9; Table 3 | STATED |
| E3 | Mean test accuracy | **98.46 %** | 1, 9; Table 3 | STATED |
| E4 | Best / worst trial | 99.31 % / 97.78 % | 9; Fig. 8 | STATED |
| E5 | Std. over trials (clean data) | Not reported (only the per-trial bars in Fig. 7) | 9 | UNKNOWN |
| E6 | Main error | Label 4 (IR007) misclassified as labels 1, 2 and 6 | 9 | STATED |
| E7 | Feature visualisation | t-SNE of raw signal, TD&FD, TD&EMD and the fusion layer (Fig. 12) | 11–13 | STATED |

### 6a. Baselines (Table 4, p.11)

| # | Parameter | Value | Page | Status |
|---|---|---|---|---|
| B1 | Classifiers | SVM, KNN, ANN, LSTM | 11 | STATED |
| B2 | Feature set 1 (TD&FD) | 15 time-domain + 3 frequency-domain features | 11 | STATED |
| B3 | Feature set 2 (TD&EMD) | 10 time-domain + 4 EMD energy features | 11, 13 | STATED |
| B4 | Feature definitions | Only cited (Chen & Li 2017; Thamba & Aravind 2018) | 13 | UNKNOWN |
| B5 | Baseline hyperparameters | Not given | — | UNKNOWN |
| B6 | Raw-signal LSTM accuracy | 66.39 % | 11, 13 | STATED |
| B7 | Best baseline accuracy | 89.72 % (SVM, TD&EMD) | 11 | STATED |

### 6b. Noise robustness (pp.13–16)

| # | Parameter | Value | Page | Status |
|---|---|---|---|---|
| N1 | Noise type | Additive white Gaussian | 13 | STATED |
| N2 | SNR definition | SNR = 10 log₁₀(Pₛ / Pₙ) dB | 16 (Eq. 17) | STATED |
| N3 | SNR levels | −2, 0, 2, 4, 6, 8, 10 dB | 14 (Table 5) | STATED |
| N4 | Noise added to train, test, or both? | Not stated | — | UNKNOWN |
| N5 | Retrained per SNR? | "Structure and parameters are not changed": ambiguous (same hyperparameters vs. same trained weights) | 16 | UNKNOWN |
| N6 | Pₛ computed per window or per file? | Not stated | — | UNKNOWN |
| N7 | Trials per SNR | 10 | 16 | STATED |
| N8 | MCNN-LSTM results | 77.27 ± 1.43 (−2 dB) → 81.41 → 85.90 → 88.19 → 90.11 → 92.77 → 95.25 ± 1.14 (10 dB) | 14 (Table 5) | STATED |
| N9 | Where it loses | At −2 dB the SVM with TD&EMD (79.12 %) beats MCNN-LSTM | 16 | STATED |

---

## 7. Reproduction risk list (all the UNKNOWNs)

Ranked by how much each can move accuracy. The "Our default" column is **our decision, not the paper's**. Write it into `configs/` and report it in the thesis as a reproduction choice.

| Rank | Unknown | Why it matters | Our default (to confirm) |
|---|---|---|---|
| 1 | W5/W6/W8 windowing + random split | Decides whether test windows overlap training data → can inflate accuracy | Non-overlapping windows; run **both** Chen's random split and a leakage-free split |
| 2 | T5 optimizer, T8 batch size | Strongly affect convergence in 400 epochs | Adam, batch 32; sweep {16, 32, 64} if we are > 3 points short |
| 3 | P3 anti-alias filter | Changes which frequencies survive down-sampling | Plain `x[::8]` (closest reading); try `scipy.signal.decimate` as an ablation |
| 4 | P5 normalisation | Affects tanh saturation | None first (matches Fig. 6); z-score per window as an ablation |
| 5 | A14 LSTM axis | Wrong axis gives a different model with a different param count | Transpose to (30, 27) so the param count matches 21 120 |
| 6 | A18 dropout rate | Overfitting control | 0.5 |
| 7 | T3 MSE loss | Reproduce as stated; cross-entropy as an ablation | MSE for reproduction |
| 8 | A15, A21, T11 | Small effects | Keras defaults; 5 fixed seeds, logged |
| 9 | N4–N6 noise protocol | Changes what "robust" means | Train clean, test noisy; Pₛ per window |
| 10 | D13 normal-file sampling rate | If it isn't 48 kHz, class 3 is on a different time scale | Verify by sample count before any loader code |

---
