from __future__ import annotations

"""Helper for processing user queries in the RAG app."""

from vasagent.ktv import predict_ktv_from_text


def answer_query(query: str, full_text: str, qa_chain, api_key: str | None = None) -> str:
    """Return an answer to ``query``.

    If the question appears to request a Kt/V prediction, the function uses
    :func:`predict_ktv_from_text` to compute the value from ``full_text``.
    Otherwise it calls ``qa_chain.invoke``.
    """

    if "ktv" in query.lower():
        try:
            result = predict_ktv_from_text(full_text, api_key=api_key)
            return f"Predicted Kt/V: {result}"
        except Exception as e:  # pragma: no cover - pass through errors
            return f"Could not predict Kt/V: {e}"

    output = qa_chain.invoke({"input": query})
    return output.get("answer", "")
