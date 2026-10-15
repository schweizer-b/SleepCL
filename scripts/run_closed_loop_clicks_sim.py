# Maintained SleepCL offline replay and simulated-click logging entry point.
# Supported scheduler: armed. No real audio output.

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import mne
import joblib
import matplotlib.pyplot as plt

from sleepcl.preproc import CausalIIR, FilterSpec
from sleepcl.stream import ChunkStreamer, StreamConfig, RingBuffer1D
from sleepcl.staging.features import extract_epoch_features
from sleepcl.labels_sleepedf import load_sleepedf_stage_epochs
from sleepcl.so_phase.detector import SOPhaseDetector, SOPhaseSpec
# EEGNet is an optional Day 11 extension; classical staging does not need Torch.
if TYPE_CHECKING:
    from sleepcl.staging.torch_eegnet import EEGNetSmall, EEGNetConfig


def _parse_stages_csv(s: str) -> set:
    parts = [p.strip() for p in str(s).split(",")]
    return {p for p in parts if p}


def _robust_z_per_channel(x: np.ndarray) -> np.ndarray:
    """Robust normalisation per channel: (x - median) / (MAD + eps). x: (C, T)."""
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    return (x - med) / (mad + 1e-6)


def _top1_and_margin(proba: np.ndarray, classes: List[str]) -> Tuple[str, float, float]:
    order = np.argsort(proba)
    top1 = int(order[-1])
    top2 = int(order[-2]) if proba.size >= 2 else top1
    stage = str(classes[top1])
    p = float(proba[top1])
    margin = float(proba[top1] - proba[top2]) if proba.size >= 2 else 0.0
    return stage, p, margin


def _load_torch_eegnet(bundle: Dict, model_path: Path) -> Tuple[EEGNetSmall, EEGNetConfig, List[str], List[str]]:
    import torch
    from sleepcl.staging.torch_eegnet import EEGNetSmall, EEGNetConfig

    cfg = EEGNetConfig(**bundle["cfg"])
    net = EEGNetSmall(cfg)

    weights_file = model_path.with_name(bundle["weights_path"])

    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    try:
        ckpt = torch.load(weights_file, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(weights_file, map_location="cpu")

    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    classes = list(bundle["classes"])
    channels = list(bundle.get("channels", []))
    return net, cfg, classes, channels


def _print_model_info(bundle: Dict, model_path: Path) -> None:
    if "model" in bundle:
        clf = bundle["model"]
        if hasattr(clf, "steps"):
            print("[MODEL] sklearn Pipeline steps:", [n for n, _ in clf.steps])
            print("[MODEL] Final estimator:", clf.steps[-1][1].__class__.__name__)
        else:
            print("[MODEL] sklearn Estimator:", clf.__class__.__name__)
    else:
        kind = bundle.get("model_kind", "unknown")
        print(f"[MODEL] kind: {kind}")
        if kind == "torch_eegnet":
            print("[MODEL] channels (bundle):", bundle.get("channels", None))
            print("[MODEL] cfg:", bundle.get("cfg", None))
            print("[MODEL] weights:", str(model_path.with_name(bundle.get("weights_path", ""))))
            print("[MODEL] classes:", bundle.get("classes", None))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--psg", required=True)
    ap.add_argument("--hyp", required=True, help="Hypnogram EDF (truth overlay/logging)")
    ap.add_argument("--model", required=True, help="Path to stager bundle .joblib")

    ap.add_argument("--ch", default="EEG Fpz-Cz", help="Primary EEG channel (also used for SO phase)")
    ap.add_argument(
        "--ch2",
        default="",
        help='Optional second EEG channel. For sklearn models: enables 2-channel gating (both must pass). '
             'For torch_eegnet: used as 2nd input channel (model is already multi-channel).',
    )
    ap.add_argument("--stages", default="N2,N3", help='Comma-separated allowed stimulation stages (default: "N2,N3")')

    ap.add_argument("--start", type=float, default=31140.0)
    ap.add_argument("--dur", type=float, default=180.0)

    ap.add_argument("--chunk_sec", type=float, default=0.05)
    ap.add_argument("--stage_update_sec", type=float, default=5.0)
    ap.add_argument("--smooth_k", type=int, default=3)

    ap.add_argument("--min_p", type=float, default=0.60)
    ap.add_argument("--min_margin", type=float, default=0.15)

    ap.add_argument("--so_window_sec", type=float, default=12.0)
    ap.add_argument("--target_phase", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--min_amp_uv", type=float, default=40.0)

    ap.add_argument("--n_clicks", type=int, default=2, choices=[1, 2])
    ap.add_argument("--train_refractory_sec", type=float, default=1.0)
    ap.add_argument("--click_latency_sec", type=float, default=0.0, help="Simulated minimum wait until another eligible SO candidate; NOT measured actuation or wall-clock latency")
    ap.add_argument("--isi_min_sec", type=float, default=1.0)
    ap.add_argument("--armed_max_wait_sec", type=float, default=2.0)

    ap.add_argument("--scheduler", default="armed", choices=["armed"], help="Only armed phase-candidate scheduling is implemented; historical fixed option is unsupported")

    ap.add_argument("--outdir", default="runs/day8_clicks")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    allowed_stages = _parse_stages_csv(args.stages)
    use_ch2 = bool(str(args.ch2).strip())
    ch2_name = str(args.ch2).strip() if use_ch2 else ""

    (outdir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("=== run_closed_loop_clicks_sim.py ===")
    print(f"PSG: {Path(args.psg).resolve()}")
    print(f"HYP: {Path(args.hyp).resolve()}")
    print(f"MODEL: {Path(args.model).resolve()}")
    print(f"OUTDIR: {outdir.resolve()}")
    print(f"SCHEDULER: {args.scheduler} | n_clicks={args.n_clicks}")
    print("[SIMULATION] click_latency_sec is a minimum wait for a later phase candidate; logged click times are EEG-sample event times, not measured stimulus/audio latency.")
    print(f"CH1 (primary/SO): {args.ch}")
    print(f"CH2: {ch2_name if use_ch2 else '(none)'}")
    print(f"ALLOWED_STAGES: {sorted(list(allowed_stages))}")

    # ---- Load model bundle ----
    model_path = Path(args.model)
    bundle = joblib.load(model_path)

    if "model" in bundle:
        model_kind = "sklearn"
    else:
        model_kind = bundle.get("model_kind", "unknown")

    if model_kind not in {"sklearn", "torch_eegnet"}:
        raise ValueError(f"Unsupported model bundle format (keys={list(bundle.keys())}).")

    # New Day10 participant-clean bundles declare an explicit compatibility contract.
    # Historical bundles lack this marker; retain their earlier behaviour for audit replay.
    if model_kind == "sklearn" and bundle.get("bundle_metadata_version") == 2:
        if str(bundle.get("channel")) != str(args.ch):
            raise ValueError(f"Model primary channel {bundle.get('channel')!r} != runner --ch {args.ch!r}")
        if not np.isclose(float(bundle.get("epoch_sec", -1)), 30.0):
            raise ValueError("This runner uses 30s stage history; incompatible bundle epoch_sec")
        cfg = bundle.get("preproc", {})
        # Runner staging filter is fixed at these values; don't silently consume
        # a model trained on a different signal-conditioning recipe.
        expected = {"notch_hz": 50.0, "notch_q": 30.0, "band_hz": [0.3, 35.0], "order": 4}
        for key, wanted in expected.items():
            got = cfg.get(key)
            if key == "band_hz":
                ok = isinstance(got, (list, tuple)) and len(got) == 2 and np.allclose(got, wanted)
            else:
                ok = got is not None and np.isclose(float(got), wanted)
            if not ok:
                raise ValueError(f"Model preprocessing mismatch for {key}: expected {wanted}, got {got}")
        fitted = bundle["model"]
        # Day10 dataset X is stored as float32. v2 online features must use
        # that same input dtype before the fitted sklearn Pipeline receives them.
        # Older v2 bundles without this explicit field used the same float32
        # Day10 contract; historical unversioned bundles keep legacy semantics.
        if bundle.get("feature_dtype", "float32") != "float32":
            raise ValueError("Model v2 feature_dtype must be float32 to match the Day10 dataset/online contract")
        declared_n = bundle.get("n_features")
        if declared_n is None or (hasattr(fitted, "n_features_in_") and int(fitted.n_features_in_) != int(declared_n)):
            raise ValueError("Model fitted input dimension differs from bundle n_features")
        if use_ch2:
            print("[MODEL] CH2 classical gate uses same single-channel estimator separately; NOT joint two-channel training")

    _print_model_info(bundle, model_path)

    # sklearn
    clf = None
    f_spec = None
    classes = None

    # torch
    torch_net: Optional[EEGNetSmall] = None
    torch_channels: List[str] = []

    if model_kind == "sklearn":
        clf = bundle["model"]
        f_spec = bundle["feature_spec"]
        final_est = clf.steps[-1][1] if hasattr(clf, "steps") else clf
        classes = list(final_est.classes_)
    else:
        import torch
        import torch.nn.functional as F

        torch_net, _, classes, torch_channels = _load_torch_eegnet(bundle, model_path)

    assert classes is not None

    # ---- Load PSG ----
    raw = mne.io.read_raw_edf(Path(args.psg), preload=True, verbose="error")
    sfreq = float(raw.info["sfreq"])

    if args.ch not in raw.ch_names:
        raise ValueError(f"Channel '{args.ch}' not found. Available: {raw.ch_names}")

    # staging channel order
    if model_kind == "torch_eegnet":
        stage_ch_names = torch_channels if torch_channels else [args.ch] + ([ch2_name] if use_ch2 else [])
    else:
        stage_ch_names = [args.ch] + ([ch2_name] if use_ch2 else [])

    for ch in stage_ch_names:
        if ch not in raw.ch_names:
            raise ValueError(f"Missing staging channel '{ch}' in PSG. Available: {raw.ch_names}")

    x_so_full = raw.get_data(picks=[args.ch]).squeeze()  # volts
    x_stage_full = raw.get_data(picks=stage_ch_names)    # (C, T) volts

    # ---- Truth labels ----
    y_true, t0 = load_sleepedf_stage_epochs(raw, args.hyp, epoch_sec=30)

    def true_stage_at(time_sec: float) -> str:
        idx = int(np.searchsorted(t0, time_sec, side="right") - 1)
        if idx < 0 or idx >= len(y_true):
            return "UNK"
        return str(y_true[idx])

    # ---- Slice segment ----
    i0 = int(round(args.start * sfreq))
    i1 = int(round((args.start + args.dur) * sfreq))
    i1 = min(i1, x_so_full.shape[0])

    x_so = x_so_full[i0:i1]
    x_stage = x_stage_full[:, i0:i1]  # (C, n_samp)

    n_samp = int(x_so.shape[0])
    t = np.arange(n_samp) / sfreq + float(args.start)

    # ---- Filters + buffers ----
    stage_filts = [
        CausalIIR(sfreq=sfreq, spec=FilterSpec(notch_hz=50.0, notch_q=30.0, band_hz=(0.3, 35.0), order=4))
        for _ in stage_ch_names
    ]
    stage_rbs = [RingBuffer1D(sfreq=sfreq, max_sec=60.0) for _ in stage_ch_names]

    so_filt = CausalIIR(
        sfreq=sfreq,
        spec=FilterSpec(notch_hz=50.0, notch_q=30.0, band_hz=(0.3, 1.2), order=4),
    )

    so_det = SOPhaseDetector(
        SOPhaseSpec(
            sfreq=sfreq,
            window_sec=float(args.so_window_sec),
            target_phase_rad=float(args.target_phase),
            tolerance_rad=float(args.tol),
            min_amp_uv=float(args.min_amp_uv),
            robust_amp=True,
            edge_detect=True,
        )
    )

    stage_update_n = max(1, int(round(args.stage_update_sec * sfreq)))
    next_stage_update = stage_update_n

    if model_kind == "sklearn" and use_ch2:
        proba_q1 = deque(maxlen=max(1, int(args.smooth_k)))
        proba_q2 = deque(maxlen=max(1, int(args.smooth_k)))
    else:
        proba_q = deque(maxlen=max(1, int(args.smooth_k)))

    current_stage = "UNK"
    current_p = np.nan
    current_margin = np.nan

    current_stage_ch2 = "NA"
    current_p_ch2 = np.nan
    current_margin_ch2 = np.nan

    so_sig_uv = np.zeros(n_samp, dtype=float)
    click_mask = np.zeros(n_samp, dtype=bool)

    stage_log = []
    trains_log = []
    clicks_log = []

    cand_total = 0
    train_starts = 0
    rej_no_stage = 0
    rej_stage = 0
    rej_conf = 0
    rej_train_refrac = 0

    clicks_total = 0
    clicks_delivered = 0
    drop_outside = 0
    drop_no_stage = 0
    drop_stage = 0
    drop_conf = 0
    drop_end_of_segment = 0

    train_active = False
    train_id = 0
    train_start_t = np.nan
    last_train_start_t = -1e9

    next_click_idx = 1
    click_ready_t = -1e9
    click_deadline_t = -1e9
    n_delivered_in_train = 0

    train_pred = "UNK"
    train_p = np.nan
    train_m = np.nan
    train_true = "UNK"

    so_scale = None  # Volts -> µV autodetect

    def _have_stage() -> bool:
        return (current_stage != "UNK") and np.isfinite(current_p) and np.isfinite(current_margin)

    def _stage_ok() -> bool:
        ok1 = current_stage in allowed_stages
        if model_kind == "sklearn" and use_ch2:
            ok2 = current_stage_ch2 in allowed_stages
            return ok1 and ok2
        return ok1

    def _conf_ok() -> bool:
        ok1 = _have_stage() and (current_p >= args.min_p) and (current_margin >= args.min_margin)
        if model_kind == "sklearn" and use_ch2:
            ok2 = (current_stage_ch2 not in {"UNK", "NA"}) and np.isfinite(current_p_ch2) and np.isfinite(current_margin_ch2)
            ok2 = ok2 and (current_p_ch2 >= args.min_p) and (current_margin_ch2 >= args.min_margin)
            return ok1 and ok2
        return ok1

    def _start_train(now_t: float):
        nonlocal train_active, train_id, train_start_t, last_train_start_t
        nonlocal next_click_idx, click_ready_t, click_deadline_t, n_delivered_in_train
        nonlocal train_pred, train_p, train_m, train_true

        train_id += 1
        train_active = True
        train_start_t = float(now_t)
        last_train_start_t = float(now_t)
        next_click_idx = 1
        n_delivered_in_train = 0

        train_pred = str(current_stage)
        train_p = float(current_p) if np.isfinite(current_p) else np.nan
        train_m = float(current_margin) if np.isfinite(current_margin) else np.nan
        train_true = true_stage_at(now_t)

        click_ready_t = float(now_t) + float(args.click_latency_sec)
        click_deadline_t = click_ready_t + float(args.armed_max_wait_sec)

    def _end_train(reason: str):
        nonlocal train_active
        trains_log.append([
            int(train_id),
            float(train_start_t),
            str(train_pred),
            float(train_p) if np.isfinite(train_p) else np.nan,
            float(train_m) if np.isfinite(train_m) else np.nan,
            str(train_true),
            int(args.n_clicks),
            int(n_delivered_in_train),
            str(reason),
        ])
        train_active = False

    # ---- Streaming ----
    streamer = ChunkStreamer(x_so, StreamConfig(sfreq=sfreq, chunk_sec=args.chunk_sec))
    sample_count = 0

    for _, so_chunk in streamer:
        so_chunk = np.asarray(so_chunk, dtype=float)
        n = int(len(so_chunk))
        if n <= 0:
            continue

        start_i = sample_count
        end_i = sample_count + n

        # Precompute causal filter outputs for this acquisition chunk. Stage
        # buffers are advanced sample-by-sample below, so inference cannot
        # consume samples later than its nominal decision timestamp.
        stage_filtered = [
            stage_filts[ci].update(np.asarray(x_stage[ci, start_i:end_i], dtype=float))
            for ci in range(len(stage_ch_names))
        ]

        # SO filter
        y_so = so_filt.update(so_chunk)

        if so_scale is None:
            medabs = float(np.nanmedian(np.abs(y_so))) if np.size(y_so) else 0.0
            # If in Volts: ~1e-6–1e-5; if already µV: ~1–10
            so_scale = 1e6 if medabs < 1e-3 else 1.0
            unit = "Volts->µV" if so_scale == 1e6 else "already µV"
            print(f"[SO] autodetect: median|x|={medabs:.3e} => {unit}")

        y_so_uv = y_so * float(so_scale)
        so_sig_uv[start_i:end_i] = y_so_uv

        sample_count = end_i

        # Process each input sample in timestamp order, even when a chunk
        # straddles multiple 5 s staging boundaries. Neither the stage buffer
        # nor its gate can look past the current candidate's sample timestamp.
        for k in range(n):
            sample_i = start_i + k
            now_t = float(args.start) + (sample_i / sfreq)
            for ci, rb in enumerate(stage_rbs):
                rb.append(stage_filtered[ci][k:k + 1])

            if sample_i + 1 >= next_stage_update:
                stage_t = now_t
                need = int(round(30.0 * sfreq))

                ep_list = []
                ok = True
                for rb in stage_rbs:
                    ep = rb.get_last(30.0)
                    if len(ep) < need:
                        ok = False
                        break
                    ep_list.append(np.asarray(ep, dtype=float))

                if ok:
                    if model_kind == "sklearn":
                        feats1 = extract_epoch_features(ep_list[0], sfreq, f_spec).reshape(1, -1)
                        if bundle.get("bundle_metadata_version") == 2:
                            feats1 = feats1.astype(np.float32)
                        proba1 = clf.predict_proba(feats1)[0]

                        if use_ch2:
                            feats2 = extract_epoch_features(ep_list[1], sfreq, f_spec).reshape(1, -1)
                            if bundle.get("bundle_metadata_version") == 2:
                                feats2 = feats2.astype(np.float32)
                            proba2 = clf.predict_proba(feats2)[0]

                            proba_q1.append(proba1)
                            proba_q2.append(proba2)
                            p1 = np.mean(np.vstack(proba_q1), axis=0)
                            p2 = np.mean(np.vstack(proba_q2), axis=0)

                            current_stage, current_p, current_margin = _top1_and_margin(p1, classes)
                            current_stage_ch2, current_p_ch2, current_margin_ch2 = _top1_and_margin(p2, classes)

                            stage_log.append([
                                stage_t, current_stage, current_p, current_margin, true_stage_at(stage_t),
                                current_stage_ch2, current_p_ch2, current_margin_ch2,
                            ])
                        else:
                            proba_q.append(proba1)
                            p = np.mean(np.vstack(proba_q), axis=0)
                            current_stage, current_p, current_margin = _top1_and_margin(p, classes)
                            stage_log.append([stage_t, current_stage, current_p, current_margin, true_stage_at(stage_t)])
                    else:
                        x_ep = np.vstack(ep_list)  # volts
                        x_uv = x_ep * 1e6
                        x_norm = _robust_z_per_channel(x_uv).astype(np.float32)

                        xt = torch.from_numpy(x_norm).unsqueeze(0)  # (1,C,T)
                        with torch.no_grad():
                            logits = torch_net(xt)
                            proba = F.softmax(logits, dim=1).cpu().numpy().squeeze()

                        proba_q.append(proba)
                        p = np.mean(np.vstack(proba_q), axis=0)

                        current_stage, current_p, current_margin = _top1_and_margin(p, classes)
                        stage_log.append([stage_t, current_stage, current_p, current_margin, true_stage_at(stage_t)])

                next_stage_update += stage_update_n

            # Apply the freshly available stage decision to this sample only
            # after the matching input sample has entered the staging buffer.
            if train_active and (now_t > click_deadline_t):
                clicks_total += 1
                drop_outside += 1
                clicks_log.append([
                    int(train_id), int(next_click_idx), float(now_t),
                    0, "armed_timeout",
                    str(current_stage),
                    float(current_p) if np.isfinite(current_p) else np.nan,
                    float(current_margin) if np.isfinite(current_margin) else np.nan,
                    np.nan, np.nan,
                    true_stage_at(now_t),
                ])
                _end_train("armed_timeout")
                # Continue with SO detection at this same sample: a new train
                # can start if this candidate passes the usual gates.

            state = so_det.update(np.asarray([y_so[k]], dtype=float))
            if state is None or (not state.get("candidate", False)):
                continue

            cand_total += 1

            have_stage = _have_stage()
            stage_ok = _stage_ok()
            conf_ok = have_stage and _conf_ok()
            train_refrac_ok = (now_t - last_train_start_t) >= float(args.train_refractory_sec)

            if not train_active:
                if not have_stage:
                    rej_no_stage += 1
                    continue
                if not stage_ok:
                    rej_stage += 1
                    continue
                if not conf_ok:
                    rej_conf += 1
                    continue
                if not train_refrac_ok:
                    rej_train_refrac += 1
                    continue

                _start_train(now_t)
                train_starts += 1

            if now_t < click_ready_t:
                continue

            have_stage_now = _have_stage()
            stage_ok_now = _stage_ok()
            conf_ok_now = have_stage_now and _conf_ok()

            if not have_stage_now:
                clicks_total += 1
                drop_no_stage += 1
                clicks_log.append([
                    int(train_id), int(next_click_idx), float(now_t),
                    0, "no_stage",
                    str(current_stage), np.nan, np.nan,
                    float(state.get("phase", np.nan)),
                    float(state.get("amp_uv", np.nan)),
                    true_stage_at(now_t),
                ])
                _end_train("no_stage_at_click")
                continue

            if not stage_ok_now:
                clicks_total += 1
                drop_stage += 1
                clicks_log.append([
                    int(train_id), int(next_click_idx), float(now_t),
                    0, "stage_gate",
                    str(current_stage),
                    float(current_p), float(current_margin),
                    float(state.get("phase", np.nan)),
                    float(state.get("amp_uv", np.nan)),
                    true_stage_at(now_t),
                ])
                _end_train("stage_gate_at_click")
                continue

            if not conf_ok_now:
                clicks_total += 1
                drop_conf += 1
                clicks_log.append([
                    int(train_id), int(next_click_idx), float(now_t),
                    0, "conf_gate",
                    str(current_stage),
                    float(current_p), float(current_margin),
                    float(state.get("phase", np.nan)),
                    float(state.get("amp_uv", np.nan)),
                    true_stage_at(now_t),
                ])
                _end_train("conf_gate_at_click")
                continue

            clicks_total += 1
            clicks_delivered += 1
            n_delivered_in_train += 1
            click_mask[sample_i] = True

            clicks_log.append([
                int(train_id), int(next_click_idx), float(now_t),
                1, "delivered",
                str(current_stage),
                float(current_p), float(current_margin),
                float(state.get("phase", np.nan)),
                float(state.get("amp_uv", np.nan)),
                true_stage_at(now_t),
            ])

            if next_click_idx >= int(args.n_clicks):
                _end_train("completed")
            else:
                next_click_idx += 1
                click_ready_t = now_t + float(args.isi_min_sec)
                click_deadline_t = click_ready_t + float(args.armed_max_wait_sec)

    if train_active:
        # At the exclusive segment-end boundary, the pending slot is cancelled;
        # this is NOT a physical/attempted click. Explicitly log the terminal
        # decision so train and click-slot accounting remains inspectable.
        segment_end_t = float(args.start) + n_samp / sfreq
        clicks_total += 1
        drop_end_of_segment += 1
        clicks_log.append([
            int(train_id), int(next_click_idx), float(segment_end_t),
            0, "end_of_segment", str(current_stage),
            float(current_p) if np.isfinite(current_p) else np.nan,
            float(current_margin) if np.isfinite(current_margin) else np.nan,
            np.nan, np.nan,
            true_stage_at(segment_end_t - 1.0 / sfreq),
        ])
        _end_train("end_of_segment")

    # ---- Save ----
    if model_kind == "sklearn" and use_ch2:
        stage_header = "time_sec,pred,p_top1,margin,true,pred_ch2,p_top1_ch2,margin_ch2"
    else:
        stage_header = "time_sec,pred,p_top1,margin,true"

    stage_rows = [stage_header]
    for r in stage_log:
        if len(r) == 8:
            stage_rows.append(f"{r[0]:.3f},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]},{r[5]},{r[6]:.4f},{r[7]:.4f}")
        else:
            stage_rows.append(f"{r[0]:.3f},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]}")
    (outdir / "stage_trace.csv").write_text("\n".join(stage_rows), encoding="utf-8")

    train_rows = ["train_id,t_start,pred_start,p_top1_start,margin_start,true_start,n_planned,n_delivered,end_reason"]
    for r in trains_log:
        train_rows.append(f"{r[0]},{r[1]:.3f},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]},{r[6]},{r[7]},{r[8]}")
    (outdir / "trains.csv").write_text("\n".join(train_rows), encoding="utf-8")

    click_rows = ["train_id,click_idx,time_sec,delivered,reason,pred,p_top1,margin,phase_rad,amp_uv,true"]
    for r in clicks_log:
        click_rows.append(
            f"{r[0]},{r[1]},{r[2]:.3f},{r[3]},{r[4]},{r[5]},{r[6]:.4f},{r[7]:.4f},{r[8]:.6f},{r[9]:.2f},{r[10]}"
        )
    (outdir / "clicks.csv").write_text("\n".join(click_rows), encoding="utf-8")

    plt.figure(figsize=(12, 4))
    plt.plot(t, so_sig_uv, label="SO-band (0.3–1.2 Hz), µV")
    plt.scatter(t[click_mask], so_sig_uv[click_mask], marker="x", label="CLICK delivered", zorder=5)
    lo, hi = np.percentile(so_sig_uv, [1, 99])
    pad = 0.2 * (hi - lo) if hi > lo else 20.0
    plt.ylim(lo - pad, hi + pad)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Delivered clicks on SO-band signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "so_clicks.png", dpi=150)
    plt.close()

    minutes = float(args.dur) / 60.0 if args.dur > 0 else np.nan
    rate = (clicks_delivered / minutes) if minutes > 0 else np.nan
    delivered_true = [r[-1] for r in clicks_log if int(r[3]) == 1]
    frac_true = float(np.mean([tl in allowed_stages for tl in delivered_true])) if len(delivered_true) else np.nan

    print("\n=== Closed-loop clicks summary ===")
    print(f"Candidates: {cand_total}")
    print(f"Train starts: {train_starts} (rej: no_stage={rej_no_stage}, stage={rej_stage}, conf={rej_conf}, train_refrac={rej_train_refrac})")
    print(f"Click decisions: total={clicks_total}, delivered={clicks_delivered}, dropped(timeout/no_stage/stage/conf/end_segment)={drop_outside}/{drop_no_stage}/{drop_stage}/{drop_conf}/{drop_end_of_segment}")
    print(f"Delivered click rate: {rate:.2f} / min")
    print(f"Fraction of delivered clicks during true allowed stages: {frac_true:.3f}")
    print(f"Saved: {(outdir / 'clicks.csv').resolve()}")
    print(f"Saved: {(outdir / 'trains.csv').resolve()}")
    print(f"Saved: {(outdir / 'stage_trace.csv').resolve()}")
    print(f"Saved: {(outdir / 'so_clicks.png').resolve()}")


if __name__ == "__main__":
    main()
