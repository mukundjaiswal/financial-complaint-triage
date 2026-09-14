from complaint_triage.llm.fake import ScriptedChatModel
from complaint_triage.tasks import Classifier, Summarizer


def test_classifier_reports_both_raw_and_parsed():
    """Both, always: the report compares them, so neither can be dropped."""
    result = Classifier(ScriptedChatModel([r"Category: credit\_card"])).predict("x")
    assert result["prediction"] == "credit_card"
    assert result["raw"] == r"Category: credit\_card"


def test_classifier_records_unreadable_output_as_none():
    result = Classifier(ScriptedChatModel(["no idea"])).predict("x")
    assert result["prediction"] is None
    assert result["raw"] == "no idea"


def test_classifier_records_the_shot_count():
    examples = (("a", "credit_card"),)
    assert (
        Classifier(ScriptedChatModel(["Category: credit_card"])).predict("x")["shots"]
        == 0
    )
    assert (
        Classifier(
            ScriptedChatModel(["Category: credit_card"]), examples=examples
        ).predict("x")["shots"]
        == 1
    )


def test_summarizer_measures_compression():
    model = ScriptedChatModel(["short summary"])
    result = Summarizer(model).predict("one two three four five six")
    assert result["summary_words"] == 2
    assert result["source_words"] == 6
