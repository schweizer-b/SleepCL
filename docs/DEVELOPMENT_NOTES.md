# Development notes

These notes describe the **important design decisions and corrections**, not every audit script or intermediate failure. They are written for technical review; the internal chronological audit is maintained separately.

## From offline analysis to a pseudo-online pipeline

The project grew from Sleep-EDF inspection and feature-based stage classification into sample-chunk replay, stateful filtering, a 30-second staging context and a slow-oscillation candidate detector. The current maintained entry point is `scripts/run_closed_loop_clicks_sim.py`. It logs the decisions of an `armed` scheduler rather than delivering sound. Earlier runners remain historical variants, not alternative validated implementations.

## Four corrections that matter when reading the code

1. **Participant identity:** Sleep-EDF night identifiers were originally treated as if they represented independent subjects. The corrected `src/sleepcl/subject_ids.py` and current dataset/training tools recognise participant-level identity and guard train/validation/test separation. Historical results remain exploratory.
2. **Causal staging timing:** the runner was corrected so a stage decision only uses EEG samples available at its declared sample time; segmentation and confidence gating are now explicit.
3. **Scheduling/accounting:** `armed` scheduler behaviour was made explicit; pending/expired simulated clicks and final train state are logged consistently. The older `fixed` scheduler is not a supported public interface.
4. **Evaluation and output safety:** current research scripts guard previous output directories, require explicit final-test unlocks where applicable, and carry source/model/split provenance. These controls prevent some accidental methodological errors but are not a substitute for independent data.

A smaller feature compatibility change replaced deprecated NumPy integration with SciPy's `trapezoid`.

## What is deliberately *not* claimed

Retrospective Hilbert comparison does not establish correct online cortical up-state targeting; a simulation timestamp is not physical stimulus onset; re-analysed subjects do not form an untouched holdout; and existing-environment checks do not prove fresh-install reproducibility.

## What to keep in the private reviewer repository

Keep durable implementation, commands needed to reproduce supported workflows, and small repeatable tests if they are extracted into a proper `tests/` suite. One-off audit collectors, temporary experiment scripts, private verification JSON, generated plots/models/data and installation hotfix packages belong in the private project archive. Briefly summarise impactful tests and fixes here and the scientific scope in `VALIDATION_AND_LIMITATIONS.md`; do not paste the full J-numbered audit trail into the README.
