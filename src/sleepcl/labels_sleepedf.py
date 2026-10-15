from __future__ import annotations

import numpy as np
import mne


SLEEPEDB_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",   # merge 3+4 into N3 (AASM-style)
    "Sleep stage R": "REM",
    "Sleep stage ?": "UNK",
    "Movement time": "MOV",
}


def load_sleepedf_stage_epochs(
    psg_raw: mne.io.BaseRaw,
    hyp_path: str,
    epoch_sec: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y: (n_epochs,) array of stage strings aligned to PSG time 0..T
      t0: (n_epochs,) epoch start times in seconds
    """
    # Read annotations from the hypnogram EDF(+)
    ann = mne.read_annotations(hyp_path)

    # Attach to PSG raw so timing is aligned using meas_date/orig_time if available
    psg_raw = psg_raw.copy()
    psg_raw.set_annotations(ann, emit_warning=False)

    sfreq = float(psg_raw.info["sfreq"])
    total_sec = psg_raw.n_times / sfreq

    n_epochs = int(np.floor(total_sec / epoch_sec))
    t0 = np.arange(n_epochs) * epoch_sec
    t_mid = t0 + epoch_sec / 2.0

    y = np.full(n_epochs, "UNK", dtype=object)

    # Fill epochs by checking which annotation interval contains the epoch midpoint
    for onset, duration, desc in zip(psg_raw.annotations.onset,
                                    psg_raw.annotations.duration,
                                    psg_raw.annotations.description):
        if desc not in SLEEPEDB_STAGE_MAP:
            continue
        label = SLEEPEDB_STAGE_MAP[desc]

        start = float(onset)
        stop = float(onset + duration)

        mask = (t_mid >= start) & (t_mid < stop)
        y[mask] = label

    return y, t0
