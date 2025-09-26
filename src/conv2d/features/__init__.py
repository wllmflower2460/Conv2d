"""Feature encoding with clear contracts."""

from __future__ import annotations

from conv2d.features.fsq_contract import (
    CodesAndFeatures,
    FSQEncoder,
    encode_fsq,
    verify_fsq_invariants,
)

__all__ = [
    "encode_fsq",
    "CodesAndFeatures",
    "FSQEncoder",
    "verify_fsq_invariants",
]