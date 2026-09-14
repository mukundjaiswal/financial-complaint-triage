"""Output normalization: the 26-point file.

Measured Micro-F1 on the zero-shot task was 0.667 scoring raw generations and
0.933 scoring the output of :func:`extract_category`. Same model, same prompt,
same data. The difference was entirely that raw generations carry escape
artefacts, surrounding prose, and near-miss label spellings that a string
comparison cannot match.

This is the single most consequential module in the project, which is why it
has the most tests.
"""

from __future__ import annotations

import re
from typing import Final

from complaint_triage.categories import ALIASES, CATEGORIES

_LABEL_ALTERNATION: Final = "|".join(CATEGORIES)
_LABELLED: Final = re.compile(
    rf"categor(?:y|ies)\s*[:\-]?\s*({_LABEL_ALTERNATION})", re.IGNORECASE
)


def normalize(text: str) -> str:
    r"""Lowercase, strip, undo escaping, and fold known aliases.

    The ``\\_`` replacement matters more than it looks: the model emits
    markdown-escaped underscores often enough that leaving them in place is a
    large share of the raw-scoring failures.
    """
    normalized = str(text).lower().strip()
    normalized = normalized.replace("\\_", "_")
    for wrong, right in ALIASES.items():
        normalized = normalized.replace(wrong, right)
    return normalized


def extract_category(text: str) -> str | None:
    """Return the canonical category in ``text``, or ``None`` if unreadable.

    Two passes, in order of confidence:

    1. A labelled match (``Category: credit_card``), which is what the prompt
       asks for.
    2. A bare canonical label anywhere in the text, for generations that answer
       correctly but ignore the requested format.

    ``None`` rather than a guess: an unreadable generation is a distinct failure
    mode from a wrong one, and collapsing them hides which is happening.
    """
    normalized = normalize(text)

    match = _LABELLED.search(normalized)
    if match:
        return match.group(1).lower()

    hits = [c for c in CATEGORIES if c in normalized]
    if len(hits) == 1:
        return hits[0]

    # Two or more distinct labels mentioned is not a decision. Treating it as
    # one would silently credit the model for a coin flip.
    return None
