"""Shared finite Defender-position grid configuration."""

from __future__ import annotations


DEFENDER_X_MIN_MAP = 5.0
DEFENDER_X_MAX_MAP = 10.0
DEFENDER_X_SPACING_MAP = 1.0
CANONICAL_DEFENDER_X_MAP = tuple(float(value) for value in range(5, 11))
DEFENDER_ACTION_COUNT = len(CANONICAL_DEFENDER_X_MAP)


__all__ = [
    "CANONICAL_DEFENDER_X_MAP",
    "DEFENDER_ACTION_COUNT",
    "DEFENDER_X_MAX_MAP",
    "DEFENDER_X_MIN_MAP",
    "DEFENDER_X_SPACING_MAP",
]
