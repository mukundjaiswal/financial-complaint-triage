"""The label set, and the aliases the model actually emits.

The alias table is evidence-driven, not speculative: each entry is a spelling
observed in real generations that is semantically correct but does not match a
canonical label. Keep it small, and add to it only from observed output.
"""

from __future__ import annotations

from typing import Final

CATEGORIES: Final[tuple[str, ...]] = (
    "credit_card",
    "retail_banking",
    "credit_reporting",
    "mortgages_and_loans",
    "debt_collection",
)

#: Observed model spellings mapped to the canonical label.
ALIASES: Final[dict[str, str]] = {
    "mortgage_and_loans": "mortgages_and_loans",
    "debt_reporting": "debt_collection",
}

UNPARSED: Final = "__unparsed__"
"""Sentinel for a generation the parser could not read.

Distinct from a wrong prediction on purpose. A wrong answer is a model problem;
an unreadable one is a harness problem, and the two need different fixes.
"""
