"""The 26-point module gets the most tests."""

import pytest

from complaint_triage.postprocess import extract_category, normalize


def test_extracts_the_requested_format():
    assert extract_category("Category: credit_card") == "credit_card"


def test_is_case_and_whitespace_insensitive():
    assert extract_category("  CATEGORY:  Retail_Banking \n") == "retail_banking"


def test_undoes_markdown_escaped_underscores():
    """A large share of raw-scoring failures were exactly this."""
    assert extract_category(r"Category: mortgages\_and\_loans") == "mortgages_and_loans"


@pytest.mark.parametrize(
    ("emitted", "canonical"),
    [
        ("mortgage_and_loans", "mortgages_and_loans"),
        ("debt_reporting", "debt_collection"),
    ],
)
def test_folds_observed_aliases(emitted, canonical):
    assert extract_category(f"Category: {emitted}") == canonical


def test_recovers_a_label_that_ignored_the_requested_format():
    assert extract_category("This belongs to debt_collection.") == "debt_collection"


def test_tolerates_punctuation_variants():
    assert extract_category("Category - credit_reporting") == "credit_reporting"


def test_labelled_match_wins_over_an_incidental_mention():
    text = "Category: credit_card (not retail_banking)"
    assert extract_category(text) == "credit_card"


def test_two_bare_labels_is_not_a_decision():
    """Picking one would credit the model for a coin flip."""
    assert extract_category("could be credit_card or retail_banking") is None


def test_returns_none_when_unreadable():
    assert extract_category("I am not sure about this one.") is None


def test_empty_generation_is_unreadable_not_wrong():
    assert extract_category("") is None


def test_normalize_is_idempotent():
    once = normalize(r"Category: Credit\_Card")
    assert normalize(once) == once


def test_normalize_does_not_invent_a_label():
    assert "credit_card" not in normalize("the customer was unhappy")
