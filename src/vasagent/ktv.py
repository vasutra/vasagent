"""Utilities for Kt/V prediction."""

from __future__ import annotations

import re
from typing import Optional, Tuple


def predict_ktv(bun: float, creatinine: float) -> float:
    """Return the product of BUN and Creatinine as a simple Kt/V estimate."""

    return bun * creatinine


def extract_lab_values(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Return BUN and Creatinine values parsed from ``text``.

    The function searches for patterns like ``"BUN: 12"`` or
    ``"Creatinine 1.2"``. If either value is missing, ``None`` is returned
    in its place.
    """

    bun = None
    creatinine = None

    bun_match = re.search(r"BUN[:\s]+([\d.]+)", text, flags=re.IGNORECASE)
    if bun_match:
        try:
            bun = float(bun_match.group(1))
        except ValueError:  # pragma: no cover - regex ensures numeric
            pass

    cre_match = re.search(r"Creatinine[:\s]+([\d.]+)", text, flags=re.IGNORECASE)
    if cre_match:
        try:
            creatinine = float(cre_match.group(1))
        except ValueError:  # pragma: no cover
            pass

    return bun, creatinine


def predict_ktv_from_text(text: str, api_key: str | None = None) -> float:
    """Return predicted Kt/V from free-form report text.

    The function first attempts to parse BUN and Creatinine values using
    :func:`extract_lab_values`. If either value is missing and ``api_key`` is
    provided, it falls back to :class:`~vasagent.mcp_agent.MCPAgent` which uses
    ChatGPT to identify the lab values.
    """

    bun, creatinine = extract_lab_values(text)
    if bun is not None and creatinine is not None:
        return predict_ktv(bun, creatinine)

    if api_key is None:
        raise ValueError("BUN and Creatinine not found in text")

    from vasagent.mcp_agent import MCPAgent

    agent = MCPAgent(api_key=api_key)
    return agent.predict(text)
