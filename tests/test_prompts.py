from complaint_triage.categories import CATEGORIES
from complaint_triage.prompts import (
    classify_prompt,
    few_shot_classify_prompt,
    summarize_prompt,
)


def test_classify_prompt_lists_every_valid_category():
    prompt = classify_prompt("my card was charged twice")
    for category in CATEGORIES:
        assert category in prompt


def test_classify_prompt_carries_the_complaint():
    assert "charged twice" in classify_prompt("my card was charged twice")


def test_classify_prompt_uses_the_instruction_format():
    prompt = classify_prompt("x")
    assert prompt.startswith("<s>[INST]")
    assert "[/INST]" in prompt


def test_few_shot_prompt_includes_every_example():
    examples = [("a complaint", "credit_card"), ("another", "debt_collection")]
    prompt = few_shot_classify_prompt("target", examples)
    assert "a complaint" in prompt
    assert "another" in prompt
    assert "target" in prompt


def test_few_shot_examples_precede_the_target():
    examples = [("EXAMPLE", "credit_card")]
    prompt = few_shot_classify_prompt("TARGET", examples)
    assert prompt.index("EXAMPLE") < prompt.index("TARGET")


def test_zero_shot_prompt_has_no_examples():
    assert few_shot_classify_prompt("x", []) == classify_prompt("x")


def test_summarize_prompt_forbids_outside_information():
    prompt = summarize_prompt("x")
    assert "Do not add assumptions" in prompt
