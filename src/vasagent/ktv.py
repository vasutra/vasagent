"""Utilities for Kt/V prediction."""

def predict_ktv(BUN: float, Creatinine: float) -> float:
    """Return the product of BUN and Creatinine as a simple Kt/V estimate."""
    return BUN * Creatinine
