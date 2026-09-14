# Method notes

## Why raw scoring was wrong, in detail

The zero-shot prompt asks for `Category: <label>`. The model complied most of
the time. The failures that cost 26 points of measured F1 were these, in
descending order of frequency:

| Failure | Example generation | Gold |
|---|---|---|
| Markdown-escaped underscores | `Category: credit\_card` | `credit_card` |
| Answer wrapped in prose | `This complaint belongs to debt_collection.` | `debt_collection` |
| Near-miss label spelling | `Category: mortgage_and_loans` | `mortgages_and_loans` |
| Semantically equivalent label | `Category: debt_reporting` | `debt_collection` |

None of these is a wrong answer. All four score zero under string comparison.

`postprocess.extract_category` handles them in two passes: a labelled match
first, because that is what the prompt asked for; then a bare canonical label
anywhere in the text, for generations that answered correctly while ignoring the
format.

## Why `None` rather than a guess

An unreadable generation returns `None`, and the metric counts it as wrong. It
is deliberately **not** dropped, because dropping it shrinks the denominator and
flatters the score. It is also not guessed at, because a wrong answer and an
unreadable one need different fixes — the first is a model or prompt problem,
the second a parser problem — and a pipeline that reports them as one number
cannot tell you which you have.

`unparseable_rate` exists so the second kind is visible on its own.

## Why two bare labels is not a decision

If a generation mentions two canonical labels with no `Category:` marker, the
parser returns `None` rather than taking the first. Taking the first would
credit the model for a coin flip roughly half the time, which inflates the score
in exactly the cases where the model was least certain.

## Why the alias table stays small

Every entry in `categories.ALIASES` is a spelling observed in real generations.
The temptation is to pre-empt spellings the model has never produced, which
grows the surface a future model change can silently exploit: a broad alias
table can map a genuinely wrong answer onto a right one and report an
improvement that did not happen.

Add to it from observed output, not from imagination.

## Why few-shot lost

0.74 against 0.93, which is the opposite of the usual expectation. Two
plausible causes, neither tested:

1. **Label prior skew.** Ten examples drawn without stratification shift the
   model's prior toward the categories that happen to appear in them.
2. **Instruction dilution.** The few-shot prefix pushes the system message far
   from the generation point, and adherence to "respond in the form
   `Category: <category>`" degrades with distance.

The second is testable directly: `unparseable_rate` should be higher on the
few-shot runs if instruction adherence is the cause. That measurement is the
obvious next experiment and it has not been run.

## What would make this rigorous

1. Commit a public dataset (CFPB) as `v1.jsonl` and switch on the CI gate.
2. Stratify the few-shot examples by class and re-measure — this alone may
   explain the negative result.
3. Report per-class F1, not only micro. Micro-F1 hides which category moved.
4. Run three seeds and report the spread. The current numbers are single-run on
   small sets.
5. Add token and cost accounting alongside latency.
