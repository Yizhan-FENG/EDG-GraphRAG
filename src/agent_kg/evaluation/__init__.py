"""Offline evaluation utilities for the A1-A4 experiment suite."""

from .a2_metrics import evaluate_a2_decisions
from .scope_guard import enforce_evidence_scope_guard

__all__ = ["evaluate_a2_decisions", "enforce_evidence_scope_guard"]

__all__ = ["evaluate_a2_decisions"]
