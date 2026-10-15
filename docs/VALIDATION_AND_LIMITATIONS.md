# Validation and limitations

**Scope.** This document describes bounded engineering checks on a previously explored Sleep-EDF dataset, not a prospectively blinded evaluation or clinical-validation study. Full internal audit traces, original recordings, model binaries and subject-level outputs are retained privately rather than copied into this repository.

## Verified engineering checks

| Check | Observed result | What it supports |
|---|---|---|
| Selected N2/N3 original-EDF/LR replay | 15 logged simulated clicks across three chunk sizes; identical selected outputs | Chunk-size consistency for one previously explored window |
| 180-second Wake-labelled original-EDF/LR replay | 89 SO candidates, 0 trains, 0 clicks at both 0.05 s and 0.20 s; 16/31 stage updates labelled N2 by the model at insufficient confidence | Confidence gating suppressed clicks in that particular Wake window |
| Synthetic scheduler/state-machine checks | Deadline, pending-click and terminal accounting cases were separately accepted in internal audit | Bounded control-logic handling, not hardware timing |
| Existing-environment software checks | 34 Python files parsed, five CLI `--help` commands passed, chunk-invariant synthetic stream and finite features | Source/runtime sanity in the audited WSL environment |

These are selected, previously explored inputs. No statistical confidence bounds, independent whole-night click rates or independent-cohort generalisation are inferred from them.

## Known unresolved scientific/engineering issues

1. **Independent sleep staging:** historical Day 10 splits were by recording/night, allowing the same participant across partitions. Newer code enforces participant-level distinctions, but historical model performance and existing model bundles are not retroactively independent.
2. **SO phase meaning:** the finite-window Hilbert endpoint applied to causally filtered Fpz–Cz EEG introduces phase-definition, filter-shift and endpoint effects. Computational target phase is not a measured cortical up-state; montage polarity/physiology need separate validation.
3. **Stimulus timing:** all clicks are simulated log events. Neither physical audio onset nor audio-driver latency or real-time deadline behaviour is measured.
4. **Model/gate calibration:** false N2 predictions were observed in a labelled Wake interval even though the configured confidence gate prevented clicks. No general no-click-during-Wake guarantee follows.
5. **Evaluation provenance:** original windows, classifier/model selections and TEST subjects were not pristine independent holdouts; historical exploratory summaries must not be relabelled as final evaluation.
6. **Distribution readiness:** a clean fresh-environment install, repository asset/privacy review, GitHub destination and licensing decisions are still outstanding. Existing WSL `pip check` reported missing declared `PyYAML` and `rich`.

## Provenance guidelines

For any future experiment, record the exact source revision, model and input checksums, participant-separated split, previously declared windows, parameters, complete event/trains/stage logs, and whether any annotations were used only for retrospective assessment. Keep generated EDF/model/data and private internal audit information out of the private reviewer repository.
