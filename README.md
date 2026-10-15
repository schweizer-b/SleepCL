# SleepCL

**A research prototype for streaming sleep-EEG analysis and simulated closed-loop stimulation.**

SleepCL replays recorded Sleep-EDF EEG in small chunks, computes features for sleep staging, estimates slow-oscillation (SO) phase candidates and logs simulated click decisions. **No audio is played, no hardware is controlled and physiological up-state targeting has not been validated.**

## Pipeline

`EDF replay → causal preprocessing → 30 s staging features → N2/N3 confidence gate → SO phase candidate → armed scheduler → CSV logs`

## Where to look

| Path | Responsibility |
|---|---|
| `scripts/run_closed_loop_clicks_sim.py` | Maintained end-to-end replay, confidence gate and simulated-click scheduler |
| `src/sleepcl/stream.py`, `preproc.py` | Chunk handling and causal filtering |
| `src/sleepcl/staging/`, `src/sleepcl/subject_ids.py` | Features, optional EEGNet code and participant identification |
| `src/sleepcl/so_phase/detector.py` | Slow-oscillation phase estimate and candidate detection |
| `scripts/day10_make_dataset.py`, `day10_train_models.py` | Classical feature dataset and model-training utilities |
| `scripts/day10_stress_test_gating.py` | Exploratory gating stress test |

The maintained runner contains the **active `armed` scheduler**. Older alternative schedulers and auxiliary prototypes are deliberately omitted from this reviewer snapshot.

## How to inspect

```bash
PYTHONPATH=src python scripts/run_closed_loop_clicks_sim.py --help
```

Run this from the repository root on Linux/WSL with the scientific dependencies available. The `src/` path makes the package importable without installing the reviewer snapshot. A fresh installation has **not** been verified. A real replay requires separately obtained paired [Sleep-EDF recordings](https://physionet.org/content/sleep-edfx/) and a compatible, trusted model bundle; neither is included. Only load trusted `joblib`, NumPy object-array, and PyTorch model artifacts.

## Evidence and limitations

Selected, previously explored N2/N3 and Wake windows showed chunk-invariant simulated event logs. These tests do **not** establish independent participant generalisation, stimulation safety or physiologically correct SO up-state timing. Historical model evaluation used night-level rather than fully person-independent splits. Real device/audio latency has not been measured.

See [Validation and limitations](docs/VALIDATION_AND_LIMITATIONS.md) and [Development notes](docs/DEVELOPMENT_NOTES.md) for bounded test evidence and the most relevant engineering corrections.

**Status:** code-review prototype, not a clinical or hardware-ready release. No license is included; access and reuse terms must be agreed separately.
