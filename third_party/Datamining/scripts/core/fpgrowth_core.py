"""FP-growth-only backend used by the Agent MVP.

This file keeps the Datamining integration surface intentionally small:
it exposes mlxtend's FP-growth functions plus the same-sensor rule filter
needed by the Agent toolbox. No alternative association-mining algorithm is
included here.
"""

from __future__ import annotations

import re

from mlxtend.frequent_patterns import association_rules, fpgrowth


def sensor_id(item: str) -> str | None:
    """Return the numeric sensor prefix from an item name such as ``241_x=HIGH``."""
    match = re.match(r"^(\d+)_", str(item))
    return match.group(1) if match else None


def is_same_sensor_rule(antecedents: tuple[str, ...], consequents: tuple[str, ...]) -> bool:
    """Filter rules where both sides only describe one numeric sensor family."""
    ids = {sensor_id(item) for item in list(antecedents) + list(consequents)}
    ids.discard(None)
    return len(ids) == 1

