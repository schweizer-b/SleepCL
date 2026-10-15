from __future__ import annotations
import numpy as np

SLEEP_STAGES = ("N1", "N2", "N3", "REM")
ALL_VALID = ("W", "N1", "N2", "N3", "REM")

def make_masks(y: np.ndarray) -> dict[str, np.ndarray]:
    """
    y: (n_epochs,) array of stage strings.
    Returns dict of boolean masks with same length as y.
    """
    y = np.asarray(y)

    # epochs that are in the main 5-stage taxonomy
    valid = np.isin(y, ALL_VALID)

    sleep = np.isin(y, SLEEP_STAGES)

    # In-bed window: from first sleep epoch to last sleep epoch
    in_bed = np.zeros_like(valid, dtype=bool)
    if sleep.any():
        first = int(np.argmax(sleep))
        last = int(len(sleep) - 1 - np.argmax(sleep[::-1]))
        in_bed[first:last + 1] = True

    # Often useful: valid epochs inside in-bed
    in_bed_valid = in_bed & valid

    # Sleep-only (exclude wake completely)
    sleep_only = sleep & valid

    return {
        "valid": valid,
        "sleep": sleep,
        "in_bed": in_bed,
        "in_bed_valid": in_bed_valid,
        "sleep_only": sleep_only,
    }
