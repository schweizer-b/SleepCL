# Build a Sleep-EDF feature dataset with participant-level identifiers.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mne
import joblib

from sleepcl.labels_sleepedf import load_sleepedf_stage_epochs
from sleepcl.labels_utils import make_masks
from sleepcl.subject_ids import participant_id
from sleepcl.preproc import CausalIIR, FilterSpec
from sleepcl.staging.features import extract_epoch_features


LABEL_ORDER = ["W", "N1", "N2", "N3", "REM"]
ALLOWED_STAGES = set(LABEL_ORDER)


def get_subject_id(rec_id: str) -> str:
    # Person-level (not night-level): SC4001E0 and SC4002E0 -> SC400.
    return participant_id(rec_id)


def find_sleepedf_pairs(root: Path) -> List[Tuple[Path, Path, str, str]]:
    """
    Returns list of (psg_path, hyp_path, rec_id, subject_id)
    Hypnogram scorer letter varies: SC4011E?-Hypnogram.edf
    """
    pairs = []
    for psg in sorted(root.glob("*-PSG.edf")):
        rec_id = psg.stem.replace("-PSG", "")  # e.g., SC4011E0
        prefix = rec_id[:-1]                  # e.g., SC4011E
        hyps = sorted(root.glob(f"{prefix}?-Hypnogram.edf"))
        if not hyps:
            print(f"[WARN] No hypnogram for {psg.name} (looked for {prefix}?-Hypnogram.edf)")
            continue
        if len(hyps) > 1:
            print(f"[WARN] Multiple hypnograms for {psg.name}; using {hyps[0].name}")
        hyp = hyps[0]
        subj = get_subject_id(rec_id)
        pairs.append((psg, hyp, rec_id, subj))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 10: build cached feature dataset from Sleep-EDF EDFs.")
    ap.add_argument("--root", default="data/raw/sleep_edf", help="Folder containing *-PSG.edf and *-Hypnogram.edf")
    ap.add_argument("--ch", default="EEG Fpz-Cz")
    ap.add_argument("--epoch_sec", type=float, default=30.0)

    ap.add_argument(
        "--spec_from_model",
        default="runs/day4_baseline/stager_lr.joblib",
        help="Load feature_spec from an existing model bundle (recommended).",
    )

    # Preproc to match your training/streaming-ish staging pipeline
    ap.add_argument("--band_lo", type=float, default=0.3)
    ap.add_argument("--band_hi", type=float, default=35.0)
    ap.add_argument("--notch_hz", type=float, default=50.0)
    ap.add_argument("--notch_q", type=float, default=30.0)
    ap.add_argument("--filt_order", type=int, default=4)

    ap.add_argument("--out", required=True, help="Output .npz path, e.g. runs/day10_dataset/features_fpzcz_epoch30.npz")
    args = ap.parse_args()

    root = Path(args.root)
    out_npz = Path(args.out)
    # Historical datasets and sidecars are immutable. Rebuild to a NEW name.
    sidecar_prefix = out_npz.with_suffix("").as_posix()
    protected_outputs = [out_npz, Path(sidecar_prefix + "_feature_spec.joblib"),
                         Path(sidecar_prefix + "_meta.json")]
    for output in protected_outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite an existing dataset artefact: {output}")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    # Load feature spec
    spec_path = Path(args.spec_from_model)
    if not spec_path.exists():
        raise FileNotFoundError(f"spec_from_model not found: {spec_path}")
    spec_bundle = joblib.load(spec_path)
    if "feature_spec" not in spec_bundle:
        raise KeyError(f"'feature_spec' not found in {spec_path}")
    feature_spec = spec_bundle["feature_spec"]

    # Save feature spec alongside dataset (joblib)
    spec_out = out_npz.with_suffix("").as_posix() + "_feature_spec.joblib"
    spec_out = Path(spec_out)
    joblib.dump(feature_spec, spec_out)

    pairs = find_sleepedf_pairs(root)
    if not pairs:
        raise RuntimeError(f"No PSG/Hyp pairs found in: {root.resolve()}")

    print("=== Day 10: make dataset ===")
    print(f"ROOT:   {root.resolve()}")
    print(f"OUT:    {out_npz.resolve()}")
    print(f"CH:     {args.ch}")
    print(f"EPOCH:  {args.epoch_sec}s")
    print(f"SPEC:   {spec_path.resolve()}")
    print(f"SPEC->  {spec_out.resolve()}")
    print(f"PAIRS:  {len(pairs)}")

    X_list = []
    y_list = []
    subj_list = []
    rec_list = []

    skipped = 0
    used_recordings = []

    for psg, hyp, rec_id, subj in pairs:
        try:
            raw = mne.io.read_raw_edf(psg, preload=False, verbose="error")
            sfreq = float(raw.info["sfreq"])

            if args.ch not in raw.ch_names:
                print(f"[SKIP] {rec_id}: channel '{args.ch}' not present")
                skipped += 1
                continue

            # Load labels aligned to PSG
            y, _t0 = load_sleepedf_stage_epochs(raw, str(hyp), epoch_sec=float(args.epoch_sec))
            y = np.asarray(y, dtype=object)

            masks = make_masks(y)
            eval_mask = masks.get("in_bed_valid", None)
            if eval_mask is None:
                print(f"[SKIP] {rec_id}: 'in_bed_valid' mask not available")
                skipped += 1
                continue

            # Load channel signal (volts)
            x = raw.get_data(picks=[args.ch]).squeeze().astype(float)

            epoch_len = int(round(float(args.epoch_sec) * sfreq))
            n_epochs_sig = len(x) // epoch_len
            n_epochs = min(len(y), n_epochs_sig)

            if n_epochs < 10:
                print(f"[SKIP] {rec_id}: too few epochs after trimming ({n_epochs})")
                skipped += 1
                continue

            x = x[: n_epochs * epoch_len]
            y = y[:n_epochs]
            eval_mask = np.asarray(eval_mask[:n_epochs], dtype=bool)

            # Causal filter to match your staging features pipeline
            filt = CausalIIR(
                sfreq=sfreq,
                spec=FilterSpec(
                    notch_hz=float(args.notch_hz),
                    notch_q=float(args.notch_q),
                    band_hz=(float(args.band_lo), float(args.band_hi)),
                    order=int(args.filt_order),
                ),
            )
            x_f = filt.update(x)

            epochs = x_f.reshape(n_epochs, epoch_len)
            idx = np.where(eval_mask)[0]
            if len(idx) < 20:
                print(f"[SKIP] {rec_id}: too few in-bed-valid epochs ({len(idx)})")
                skipped += 1
                continue

            kept_here = 0
            for ei in idx:
                lbl = str(y[ei])
                if lbl not in ALLOWED_STAGES:
                    continue
                feats = extract_epoch_features(epochs[ei], sfreq, feature_spec)
                X_list.append(feats)
                y_list.append(lbl)
                subj_list.append(subj)
                rec_list.append(rec_id)
                kept_here += 1

            print(f"[OK] {rec_id}: subj={subj} | epochs_used={kept_here} | sfreq={sfreq}")
            used_recordings.append(rec_id)

        except Exception as e:
            print(f"[SKIP] {rec_id}: error: {e}")
            skipped += 1

    if not X_list:
        raise RuntimeError("No epochs collected. Check channel name, hypnogram pairing, and masks.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=object)
    subject = np.asarray(subj_list, dtype=object)
    rec_id = np.asarray(rec_list, dtype=object)

    # Save dataset
    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        subject=subject,
        rec_id=rec_id,
    )

    # Save meta
    meta = {
        "root": str(root),
        "channel": args.ch,
        "epoch_sec": float(args.epoch_sec),
        "spec_from_model": str(spec_path),
        "feature_spec_joblib": spec_out.name,
        "preproc": {
            "notch_hz": float(args.notch_hz),
            "notch_q": float(args.notch_q),
            "band_hz": [float(args.band_lo), float(args.band_hi)],
            "order": int(args.filt_order),
        },
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_dtype": "float32",  # Cached X and online v2 runner must use the same dtype.
        "n_recordings_used": int(len(set(used_recordings))),
        "recordings_used": sorted(list(set(used_recordings))),
        "skipped_recordings": int(skipped),
        "label_counts": {k: int(np.sum(y == k)) for k in LABEL_ORDER},
        "subject_counts": int(len(set(subject.tolist()))),
        "subject_id_semantics": "sleepedf-person-v2",
        "person_ids": sorted(set(subject.tolist())),
    }
    meta_path = out_npz.with_suffix("").as_posix() + "_meta.json"
    meta_path = Path(meta_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSaved dataset:", out_npz.resolve())
    print("Saved spec   :", spec_out.resolve())
    print("Saved meta   :", meta_path.resolve())
    print("\nLabel counts :", meta["label_counts"])
    print("Subjects     :", meta["subject_counts"])


if __name__ == "__main__":
    main()
