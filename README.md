# SleepCL

**Sleep staging and simulated closed-loop auditory stimulation using EEG**

SleepCL is a Python research project exploring how automated sleep-stage classification can guide phase-targeted auditory stimulation during non-REM sleep. It replays recordings from the [Sleep-EDF dataset](https://physionet.org/content/sleep-edfx/) as an incoming EEG stream, combines stage predictions with slow-oscillation (SO) detection, and records simulated click decisions.

The project examines both **sleep-staging performance** and how prediction confidence, EEG characteristics and timing rules affect a closed-loop stimulation controller.

## Pipeline

`Sleep-EDF replay → Causal preprocessing → Sleep staging → N2/N3 confidence gate → SO phase candidates → Armed click scheduler → Results`

### 1. Sleep staging

The classifier distinguishes **Wake, N1, N2, N3 and REM** using the most recent 30 seconds of EEG. Sleep-EDF hypnogram annotations provide reference labels for training and evaluation.

- **Preprocessing:** EEG channel selection and causal filtering (0.3–35 Hz for staging).
- **Features:** seven measurements per window: Welch-based log-relative power in the delta, theta, alpha, sigma and beta bands, plus signal standard deviation and mean absolute sample-to-sample difference.
- **Models:** Logistic Regression (LR), Support Vector Machines (SVM), Random Forest (RF) and a Multilayer Perceptron (MLP), implemented with scikit-learn.
- **Evaluation:** classification metrics and confusion matrices, including participant-separated development validation.

An **EEGNet-inspired convolutional network** was also explored as a way to learn features directly from EEG. Its architecture and optional inference path are included, but this remains an experimental extension rather than a completed, comparable benchmark.

### 2. Streaming and closed-loop simulation

Recorded EEG is replayed in short chunks (**50 ms by default**). The classifier uses a rolling 30-second window, with stage probabilities updated every **5 seconds** by default and smoothed across successive predictions.

The stimulation logic combines two checks:

1. **Sleep-stage gate:** the predicted stage must be N2 or N3, with sufficient top-class probability and separation from the second-highest probability. An optional Pz–Oz channel adds an agreement gate: the same single-channel classical model evaluates both channels separately, and both must pass the criteria.
2. **SO timing:** the Fpz–Cz signal is filtered at **0.3–1.2 Hz**. A trailing-window Hilbert phase estimate and amplitude threshold identify candidate moments near a selected target phase.

An **armed scheduler** then manages one- or two-click trains, minimum intervals, refractory periods and expiry times. The simulation logs stage estimates, train and click decisions, and event markers on the SO waveform.

## Getting started

From the repository root, install the project in a Python environment:

```bash
python -m pip install -e .
```

**Train classical staging models** using an existing feature dataset and participant split:

```bash
python scripts/day10_train_models.py \
  --dataset path/to/features.npz \
  --split_json path/to/subject_split.json \
  --models lr svm_lin rf mlp \
  --outdir runs/models
```

**Run a representative closed-loop simulation** with a compatible trained model and paired Sleep-EDF files:

```bash
python scripts/run_closed_loop_clicks_sim.py \
  --psg path/to/SC4001E0-PSG.edf \
  --hyp path/to/SC4001EC-Hypnogram.edf \
  --model runs/models/mlp/stager.joblib \
  --start 32490 --dur 180 \
  --outdir runs/example
```

The selected start time is an example for that recording; predictions and simulated clicks depend on the trained model and settings. Outputs include `stage_trace.csv`, `trains.csv`, `clicks.csv` and `so_clicks.png`.

Recordings, trained model bundles and participant split files are not included. To build a new feature dataset from EDF recordings, `day10_make_dataset.py` currently requires a compatible existing model bundle supplying its feature specification. Each script supports `--help` for its options.

## Code structure

| Location | Contents |
|---|---|
| `src/sleepcl/staging/` | EEG feature extraction and exploratory EEGNet model |
| `src/sleepcl/preproc.py`, `src/sleepcl/stream.py` | Causal filtering, buffers and EEG replay |
| `src/sleepcl/so_phase/detector.py` | SO phase estimation and candidate detection |
| `scripts/day10_make_dataset.py` | Feature dataset generation |
| `scripts/day10_train_models.py` | Classical-model training and validation |
| `scripts/run_closed_loop_clicks_sim.py` | Streaming staging, gating and simulated-click scheduler |
| `scripts/day10_stress_test_gating.py` | Exploratory model/gating comparisons |

## Technical overview

For the methods, model comparisons, development decisions, phase calibration, stimulation logic and interpreted figures, see the **[SleepCL Technical Overview](docs/SleepCL_Technical_Overview.pdf)**.


