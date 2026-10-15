
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import mne

from sleepcl.labels_sleepedf import load_sleepedf_stage_epochs
from sleepcl.subject_ids import participant_id, validate_split

ALLOWED = {"N2", "N3"}


def find_sleepedf_pairs(root: Path):
    pairs = []
    for psg in sorted(root.glob("*-PSG.edf")):
        base = psg.stem.replace("-PSG", "")
        rec_id = base
        prefix = base[:-1]
        hyps = sorted(root.glob(f"{prefix}?-Hypnogram.edf"))
        if not hyps:
            print(f"[WARN] No hypnogram for {psg.name} (looked for {prefix}?-Hypnogram.edf)")
            continue
        if len(hyps) > 1:
            print(f"[WARN] Multiple hypnograms for {psg.name}; using {hyps[0].name}")
        pairs.append((psg, hyps[0], rec_id))
    return pairs


def pick_start_times(
    psg: Path,
    hyp: Path,
    dur_sec: float,
    n_windows: int,
    seed: int,
    mode: str,
    stages: List[str],
    disjoint: bool = False,
) -> List[float]:
    raw = mne.io.read_raw_edf(psg, preload=False, verbose="error")
    length_sec = raw.n_times / float(raw.info["sfreq"])
    y, t0 = load_sleepedf_stage_epochs(raw, str(hyp), epoch_sec=30)

    y = np.asarray(y, dtype=object)
    t0 = np.asarray(t0, dtype=float)

    ok = np.where(t0 + float(dur_sec) <= float(length_sec))[0]
    if ok.size == 0:
        return []

    rng = np.random.default_rng(int(seed))

    if disjoint:
        # Opt-in selection for a new predeclared protocol. Historical starts
        # and the normal CLI default are intentionally unchanged.
        selected: list[float] = []
        seen: set[float] = set()
        def consider(indices):
            for idx in rng.permutation(np.asarray(indices, dtype=int)):
                start = float(t0[int(idx)])
                if start not in seen and all(abs(start - prev) >= float(dur_sec) - 1e-8 for prev in selected):
                    selected.append(start)
                    seen.add(start)
                    if len(selected) >= n_windows:
                        return
        if mode == "stratified":
            per = int(np.ceil(n_windows / max(1, len(stages))))
            for st in stages:
                if len(selected) >= n_windows:
                    break
                # Stage preferences influence start-point sampling only.
                candidates = rng.permutation(ok[y[ok] == st])
                added = 0
                for ix in candidates:
                    if added >= per or len(selected) >= n_windows:
                        break
                    before = len(selected)
                    consider([ix])
                    added += len(selected) - before
        if len(selected) < n_windows:
            consider(ok)
        if len(selected) != n_windows:
            raise ValueError(f"Disjoint selection has only {len(selected)} of {n_windows} requested windows; choose fewer or longer recordings")
        return selected

    if mode == "random":
        idx = rng.choice(ok, size=min(n_windows, ok.size), replace=(ok.size < n_windows))
        return [float(t0[i]) for i in idx]

    # stratified: try to sample ~equal windows from each requested stage
    chosen = []
    per = int(np.ceil(n_windows / max(1, len(stages))))
    for st in stages:
        m = ok[y[ok] == st]
        if m.size == 0:
            continue
        k = min(per, m.size)
        idx = rng.choice(m, size=k, replace=False)
        chosen.extend([float(t0[i]) for i in idx])

    # fill remainder randomly
    if len(chosen) < n_windows:
        remaining = n_windows - len(chosen)
        idx = rng.choice(ok, size=remaining, replace=(ok.size < remaining))
        chosen.extend([float(t0[i]) for i in idx])

    return chosen[:n_windows]


def run_window(
    runner: Path,
    psg: Path,
    hyp: Path,
    model: Path,
    ch: str,
    start_sec: float,
    dur_sec: float,
    n_clicks: int,
    scheduler: str,
    target_phase: float,
    tol: float,
    outdir: Path,
    extra_args: List[str],
):
    cmd = [
        sys.executable, str(runner),
        "--psg", str(psg),
        "--hyp", str(hyp),
        "--model", str(model),
        "--ch", str(ch),
        "--start", str(start_sec),
        "--dur", str(dur_sec),
        "--n_clicks", str(n_clicks),
        "--scheduler", str(scheduler),
        "--target_phase", str(target_phase),
        "--tol", str(tol),
        "--outdir", str(outdir),
    ] + extra_args
    subprocess.run(cmd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarise_clicks(clicks_csv: Path, dur_sec: float) -> Dict[str, float]:
    """Count DELIVERED simulated clicks only; non-delivery rows are not clicks.

    Zero delivered means zero true/false *rates*, but true fraction is undefined.
    Missing true-stage metadata for a positive delivered count leaves stage rates
    undefined and must not be mistaken for zero false clicks.
    """
    if not (np.isfinite(dur_sec) and float(dur_sec) > 0):
        raise ValueError("dur_sec must be finite and positive")
    df = pd.read_csv(clicks_csv)
    if "delivered" not in df.columns:
        raise ValueError(f"Missing required delivered column: {clicks_csv}")
    delivery = pd.to_numeric(df["delivered"], errors="coerce")
    if not delivery.isin([0, 1]).all():
        raise ValueError("Click delivered indicators must be 0 or 1")
    df_del = df.loc[delivery == 1]
    delivered = int(len(df_del))
    minutes = float(dur_sec) / 60.0
    true_count = false_count = 0
    fraction = float("nan")
    true_rate = false_rate = float("nan")
    if delivered == 0:
        true_rate = false_rate = 0.0
    elif "true" in df_del and df_del["true"].notna().all() and (df_del["true"].astype(str).str.strip() != "").all():
        true_count = int(df_del["true"].astype(str).isin(ALLOWED).sum())
        false_count = delivered - true_count
        fraction = true_count / delivered
        true_rate = true_count / minutes
        false_rate = false_count / minutes

    return {
        "delivered_clicks": delivered,
        "delivered_rate_per_min": delivered / minutes,
        "frac_delivered_true_N2N3": fraction,
        "true_click_rate_per_min": true_rate,
        "false_click_rate_per_min": false_rate,
        "true_delivered_clicks": float(true_count) if math.isfinite(true_rate) else float("nan"),
        "false_delivered_clicks": float(false_count) if math.isfinite(false_rate) else float("nan"),
        "undelivered_decisions": int((delivery == 0).sum()),
        "exposure_min": minutes,
    }


def aggregate_stress_windows(df: pd.DataFrame) -> dict:
    """Pooled rates have sampled-window-minutes denominators, NOT click weights.

    Overlapping source-time windows are separate simulation exposures. Neither
    these rates nor the union-duration calculation estimate a whole-night dose.
    """
    if df.empty:
        raise ValueError("No windows to aggregate")
    exposure = float(df["exposure_min"].sum())
    if not (exposure > 0 and math.isfinite(exposure)):
        raise ValueError("Invalid evaluated exposure")
    n_del = int(df["delivered_clicks"].sum())
    result = {
        "window_count": int(len(df)),
        "missing_truth_windows": int(df["true_click_rate_per_min"].isna().sum()),
        "zero_click_windows": int((df["delivered_clicks"] == 0).sum()),
        "sampled_window_exposure_min": exposure,
        "delivered_clicks": n_del,
        "pooled_delivered_per_min": n_del / exposure,
        "macro_delivered_rate_per_min": float(df["delivered_rate_per_min"].mean()),
        "macro_true_click_rate_per_min": float(df["true_click_rate_per_min"].mean()),
        "macro_false_click_rate_per_min": float(df["false_click_rate_per_min"].mean()),
        "macro_fraction_true_on_nonempty_windows": float(df["frac_delivered_true_N2N3"].mean()),
        "pooled_fraction_true_of_delivered": float("nan"),
        "pooled_true_per_min": float("nan"),
        "pooled_false_per_min": float("nan"),
    }
    if df["true_click_rate_per_min"].notna().all() and df["false_click_rate_per_min"].notna().all():
        true_count = int(df["true_delivered_clicks"].sum())
        false_count = int(df["false_delivered_clicks"].sum())
        assert true_count + false_count == n_del
        result.update(pooled_fraction_true_of_delivered=(true_count / n_del if n_del else float("nan")),
                      pooled_true_per_min=true_count / exposure,
                      pooled_false_per_min=false_count / exposure,
                      true_delivered_clicks=true_count,
                      false_delivered_clicks=false_count)
    else:
        result["warning"] = "Missing click-time truth: pooled true/false click rates unavailable"
    intervals: dict[str, list[tuple[float, float]]] = {}
    for _, row in df.iterrows():
        start = float(row["start_sec"])
        dur = float(row["exposure_min"]) * 60.
        intervals.setdefault(str(row["rec_id"]), []).append((start, start + dur))
    union_seconds = 0.
    pairs_overlapping = 0
    for spans in intervals.values():
        spans.sort()
        for i, (a, b) in enumerate(spans):
            pairs_overlapping += sum(a < d and b > c for c, d in spans[i + 1:])
        last = None
        for a, b in spans:
            if last is None or a >= last:
                union_seconds += b - a
                last = b
            elif b > last:
                union_seconds += b - last
                last = b
    result.update(unique_source_coverage_min=union_seconds / 60.,
                  overlapping_window_pairs=pairs_overlapping,
                  repeat_exposure_min=exposure - union_seconds / 60.)
    return result


def ensure_new_output_directory(path: Path) -> None:
    """Never trust old clicks.csv as cache and never overwrite H1 outputs."""
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"Stress output already exists or is nonempty: {path}; use a NEW --outdir (no silent cache/overwrite)")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/sleep_edf")
    ap.add_argument("--runner", default="scripts/run_closed_loop_clicks_sim.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--ch", default="EEG Fpz-Cz")
    ap.add_argument("--dur", type=float, default=180.0)
    ap.add_argument("--n_windows", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_clicks", type=int, default=1, choices=[1, 2])
    ap.add_argument("--scheduler", default="armed", choices=["armed"], help="Only armed scheduling is implemented by the maintained runner")
    ap.add_argument("--target_phase", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=0.25)

    ap.add_argument("--split_json", default=None)
    ap.add_argument("--subset", default="all", choices=["all", "train", "validation", "test"])
    ap.add_argument("--final_test_approved", action="store_true", help="Explicit protocol-unlock for final TEST evaluation; no automatic claim that TEST persons are historically untouched")

    ap.add_argument("--start_mode", default="stratified", choices=["random", "stratified"])
    ap.add_argument("--stages", default="W,N1,REM,N2,N3", help="Comma list of stages used to select STARTS only; the full window can contain other stages")
    ap.add_argument("--disjoint_windows", action="store_true", help="Opt-in within-recording non-overlapping starts; fails if requested number cannot be selected; not independent subjects")

    ap.add_argument("--overwrite", action="store_true", help="Deprecated and refused: create a fresh output directory to preserve historical results")

    # pass-through args to runner (put last)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    if args.overwrite:
        raise ValueError("--overwrite is disabled to protect historical output; use a NEW --outdir")
    # Keep the controlling runner arguments in one place. Forwarded --extra
    # must not silently change input identity or the configuration fingerprint.
    protected_runner_flags = {"--psg", "--hyp", "--model", "--ch", "--start", "--dur", "--n_clicks", "--scheduler", "--target_phase", "--tol", "--outdir"}
    for flag in args.extra:
        if any(flag == x or flag.startswith(x + "=") for x in protected_runner_flags):
            raise ValueError(f"Conflicting runner control in --extra: {flag}; set it with the top-level CLI")
    if not (np.isfinite(args.dur) and args.dur > 0 and args.n_windows > 0):
        raise ValueError("--dur and --n_windows must be positive")
    # Fail before loading recordings or executing any runner if this would reuse
    # or replace outputs from a previous experiment.
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        raise FileExistsError(f"Stress output already exists or is nonempty: {outdir}; use a NEW --outdir")

    pairs = find_sleepedf_pairs(root)

    # Strict participant-level split for any claimed development/validation/test subset.
    if args.subset != "all" and not args.split_json:
        raise ValueError("A participant-level --split_json is mandatory for a named subset")
    if args.subset == "test" and not args.final_test_approved:
        raise ValueError("Final TEST is locked. Freeze the experiment protocol and pass --final_test_approved explicitly.")
    if args.split_json:
        split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
        groups = validate_split(split)  # Refuses historical night IDs/overlap.
        group_key = {"train": "train_subjects", "validation": "validation_subjects", "test": "test_subjects"}
        keep = groups[group_key[args.subset]] if args.subset in group_key else set.union(*groups.values())
        # A model's version-2 split must match before final TEST results are produced.
        if args.subset == "test":
            import joblib
            bundle = joblib.load(args.model)  # Local trusted model artefacts only.
            bsplit = bundle.get("split", {}) if isinstance(bundle, dict) else {}
            if (bundle.get("bundle_metadata_version") != 2 or
                bsplit.get("id_semantics") != "sleepedf-person-v2" or
                any(set(bsplit.get(k, [])) != groups[k] for k in group_key.values())):
                raise ValueError("Final TEST requires a version-2 participant-disjoint model bundle with matching frozen split")
        before = len(pairs)
        pairs = [(psg, hyp, rec_id) for (psg, hyp, rec_id) in pairs if participant_id(rec_id) in keep]
        if not pairs:
            raise ValueError(f"No recordings available for subset={args.subset}")
        print(f"[INFO] participant-disjoint subset={args.subset}: kept {len(pairs)}/{before} recordings")
    else:
        print("[WARN] No participant split supplied: exploratory ALL-data diagnostic, not a holdout evaluation")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    if not pairs:
        raise ValueError(f"No recordings available for subset={args.subset}")

    # Fingerprint every input used in new stress runs. Expensive raw-file hashes
    # are calculated once per paired recording, NOT once per window.
    # This is a manifest for new runs, not historical provenance recovery.
    runner_path = Path(args.runner).resolve(strict=True)
    model_path = Path(args.model).resolve(strict=True)
    split_path = Path(args.split_json).resolve(strict=True) if args.split_json else None
    provenance = {
        "schema": "sleepcl-stress-input-provenance-v1",
        "description": "New simulation runs only; SHA binds input BYTES, not a verified historical commit or physiological validity",
        "wrapper": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
        "model": {"path": str(model_path), "sha256": sha256_file(model_path),
                  "warning": "External torch weights, if referenced by a legacy model bundle, are not fingerprinted"},
        "split": ({"path": str(split_path), "sha256": sha256_file(split_path)} if split_path else None),
        "configuration": {
            "root": str(root.resolve()), "ch": args.ch, "dur_sec": args.dur,
            "n_windows_requested_per_recording": args.n_windows, "seed": args.seed,
            "n_clicks": args.n_clicks, "scheduler": args.scheduler,
            "target_phase": args.target_phase, "tol": args.tol,
            "subset": args.subset, "start_mode": args.start_mode,
            "stages_at_start": stages, "disjoint_windows": bool(args.disjoint_windows), "extra_args": list(args.extra),
            "final_test_approved": bool(args.final_test_approved),
        },
        "recordings": [],
        "windows": [],
    }
    rows = []
    planned = []
    for psg, hyp, rec_id in pairs:
        starts = pick_start_times(
            psg=psg,
            hyp=hyp,
            dur_sec=float(args.dur),
            n_windows=int(args.n_windows),
            seed=int(args.seed),
            mode=str(args.start_mode),
            stages=stages,
            disjoint=bool(args.disjoint_windows),
        )
        if not starts:
            print(f"[SKIP] {rec_id}: no valid start times")
            continue
        provenance["recordings"].append({
            "rec_id": rec_id, "participant_id": participant_id(rec_id),
            "psg": {"path": str(psg.resolve()), "sha256": sha256_file(psg)},
            "hyp": {"path": str(hyp.resolve()), "sha256": sha256_file(hyp)},
        })
        for wi, start in enumerate(starts):
            run_name = f"{rec_id}_win{wi:02d}_t{int(start)}"
            planned.append((psg, hyp, rec_id, run_name, float(start)))
            provenance["windows"].append({"run_name": run_name, "rec_id": rec_id, "start_sec": float(start),
                                          "duration_sec": float(args.dur), "start_stage_only": True})
    if not planned:
        raise RuntimeError("No valid windows selected; no output created")
    if len({x[3] for x in planned}) != len(planned):
        raise ValueError("Duplicate selected stress window run names; refusing an ambiguous or overwritten result")

    # Exclusive output and manifest creation. A failed run leaves an incomplete
    # directory that MUST NOT be reused. The correct action is a new --outdir.
    ensure_new_output_directory(outdir)
    (outdir / "stress_run_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for psg, hyp, rec_id, run_name, start in planned:
        run_dir = outdir / run_name
        if run_dir.exists():
            raise FileExistsError(f"Window directory collision: {run_dir}")
        run_dir.mkdir()
        run_window(
            runner=runner_path, psg=psg, hyp=hyp, model=model_path,
            ch=str(args.ch), start_sec=start, dur_sec=float(args.dur),
            n_clicks=int(args.n_clicks), scheduler=str(args.scheduler),
            target_phase=float(args.target_phase), tol=float(args.tol),
            outdir=run_dir, extra_args=list(args.extra),
        )
        required = ["run_config.json", "stage_trace.csv", "trains.csv", "clicks.csv"]
        missing = [f for f in required if not (run_dir / f).is_file()]
        if missing:
            raise RuntimeError(f"Runner did not produce complete required logs for {run_name}: {missing}")
        met = summarise_clicks(run_dir / "clicks.csv", dur_sec=float(args.dur))
        rows.append({"rec_id": rec_id, "run_name": run_name, "start_sec": start, **met})

    df = pd.DataFrame(rows)
    out_csv = outdir / "stress_gating_summary.csv"
    df.to_csv(out_csv, index=False)
    summary = aggregate_stress_windows(df)
    (outdir / "stress_aggregate.json").write_text(json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                 for k, v in summary.items()}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_csv.resolve()}")
    print(f"Saved: {(outdir / 'stress_run_manifest.json').resolve()}")
    print(f"Saved: {(outdir / 'stress_aggregate.json').resolve()}")
    print("\n=== Pooled across all sampled-window exposure (NOT unique-night time) ===")
    print(f"Delivered: {summary['pooled_delivered_per_min']:.6f}/min; "
          f"true: {summary['pooled_true_per_min']:.6f}/min; "
          f"false: {summary['pooled_false_per_min']:.6f}/min")
    print(f"Zero-click windows: {summary['zero_click_windows']}/{summary['window_count']}; "
          f"source-time overlap: {summary['repeat_exposure_min']:.2f} min")
    print("=== Per-window macro mean rates (zero-click windows included) ===")
    print(df[["delivered_rate_per_min", "true_click_rate_per_min", "false_click_rate_per_min"]].mean())
    print(f"Pooled fraction among delivered clicks: {summary['pooled_fraction_true_of_delivered']:.6f}")


if __name__ == "__main__":
    main()
