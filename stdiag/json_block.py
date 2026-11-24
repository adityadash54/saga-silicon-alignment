"""Find the fenced ```json object that ends an agent reply.

Used by the triage (findings block) and the expert dialogue (status block).
A reply may contain earlier ```json blocks (examples, quoted output), and the
final object may itself contain ``` inside a string, so the fences are tried
from the last one backwards until one parses as a JSON object that runs to
the end of the reply.
"""

from __future__ import annotations

import json

FENCE = "```json"


def final_json_block(text: str) -> tuple[int | None, dict | None]:
    """Return (start of the opening fence, parsed object), or (None, None)."""
    tail = text.rstrip()
    if not tail.endswith("```"):
        return None, None
    end = len(tail) - 3
    start = tail.rfind(FENCE, 0, end)
    while start != -1:
        try:
            parsed = json.loads(tail[start + len(FENCE):end])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return start, parsed
        start = tail.rfind(FENCE, 0, start)
    return None, None
