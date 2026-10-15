"""Sleep-EDF participant identity and strict Phase-4 partition guards.

The cassette/telemetry recording's penultimate numeric digit is NIGHT,
not an independent participant. Historical *night* identifiers must fail closed.
"""
from __future__ import annotations

import re
from typing import Mapping, Iterable

_RECORD = re.compile(r"^(SC|ST)(\d{3})\d[A-Z]\d$")
_PERSON = re.compile(r"^(SC|ST)\d{3}$")
_GROUPS = ("train_subjects", "validation_subjects", "test_subjects")


def participant_id(recording_id: str) -> str:
    """Return person ID, e.g. SC4001E0 and SC4002E0 -> SC400.

    Reject unfamiliar identifiers rather than silently creating a leaky split.
    """
    match = _RECORD.fullmatch(str(recording_id))
    if match is None:
        raise ValueError(f"Unsupported Sleep-EDF recording ID {recording_id!r}; expected SC/ST + 3 participant digits + night/scorer fields")
    return match.group(1) + match.group(2)


def validate_split(split: Mapping, available: Iterable[str] | None = None) -> dict[str, set[str]]:
    """Require nonempty, canonical participant-only and disjoint TRAIN/VAL/TEST.

    This validates group independence, *not* whether a person was previously
    inspected/tuned on during the historical project.
    """
    groups: dict[str, set[str]] = {}
    for key in _GROUPS:
        values = split.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list of participant IDs")
        if any(not isinstance(v, str) or _PERSON.fullmatch(v) is None for v in values):
            raise ValueError(f"{key} contains a night/recording/noncanonical ID; use SC400 (person), never SC4001 (night)")
        if len(values) != len(set(values)):
            raise ValueError(f"{key} repeats an identifier")
        groups[key] = set(values)
    for i, a in enumerate(_GROUPS):
        for b in _GROUPS[i + 1:]:
            overlap = groups[a] & groups[b]
            if overlap:
                raise ValueError(f"Participant overlap {a}/{b}: {sorted(overlap)}")
    if available is not None:
        observed = set(available)
        if any(_PERSON.fullmatch(str(s)) is None for s in observed):
            raise ValueError("Dataset contains noncanonical subject values; rebuild feature cache with corrected participant IDs")
        declared = set.union(*groups.values())
        if observed != declared:
            raise ValueError(f"Split/dataset subjects differ: unassigned={sorted(observed - declared)} absent={sorted(declared - observed)}")
    return groups
