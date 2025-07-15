from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - openai may not be installed
    OpenAI = None  # type: ignore

from vasagent.ktv import extract_lab_values, predict_ktv


class MCPAgent:
    """Agentic helper that leverages ChatGPT to predict Kt/V."""

    def __init__(self, api_key: str | None = None) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required for MCPAgent")
        self.client = OpenAI(api_key=api_key)

    def predict(self, text: str) -> float:
        """Return predicted Kt/V for ``text`` using ChatGPT for extraction."""
        bun, creatinine = extract_lab_values(text)
        if bun is None or creatinine is None:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract BUN and Creatinine values from the report. "
                        "Respond strictly as 'BUN:<value> Creatinine:<value>'."
                    ),
                },
                {"role": "user", "content": text},
            ]
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=messages
            )
            reply = response.choices[0].message.content or ""
            bun2, creatinine2 = extract_lab_values(reply)
            bun = bun if bun is not None else bun2
            creatinine = creatinine if creatinine is not None else creatinine2

        if bun is None or creatinine is None:
            raise ValueError("BUN or Creatinine could not be determined")

        return predict_ktv(bun, creatinine)
