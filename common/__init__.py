"""Shared deterministic randomness and error types."""

from .errors import SigningFailure
from .rng import ShakeRNG

__all__ = ["ShakeRNG", "SigningFailure"]
