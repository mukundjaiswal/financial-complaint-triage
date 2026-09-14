[![CI](https://github.com/mukundjaiswal/financial-complaint-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/mukundjaiswal/financial-complaint-triage/actions/workflows/ci.yml)

# Financial Complaint Triage

Routing free-text consumer complaints into product categories and summarizing
them, using a quantized open-weight instruction model with **no fine-tuning**.

Two things are compared here: two prompting methods, and two ways of scoring the
same output. The second mattered more.

This is a personal project. It is not derived from any employer's systems, data
or code.

---

## The result this project exists for

| Approach | Micro-F1 |
|---|---|
| Zero-shot, scoring the **raw** generation | **0.67** |
| Zero-shot, scoring after **output normalization** | **0.93** |
| Few-shot (10 in-context examples), normalized | **0.74** |

Same model. Same prompt. Same data. **The 26-point gap is entirely the layer
that reads the model's answer.**

### Two comparisons, and the surprising one is not the method

| What changed | Effect on measured F1 |
|---|---|
| Zero-shot to few-shot - **the method** | **-19 points** (0.93 to 0.74) |
| Raw to normalized scoring - **the measurement** | **+26 points** (0.67 to 0.93) |

The measurement layer moved the number more than the method change did, and in
the opposite direction from what anyone would predict. That is the finding.
Measured quality was a property of the pipeline, not of the model.

The model was already choosing the right category most of the time. What it was
not doing was saying so in a form a string comparison could match: markdown-
escaped underscores (`credit\_card`), answers wrapped in prose, and near-miss
label spellings like `mortgage_and_loans` for `mortgages_and_loans`.

That is the whole argument for a harness. A pipeline without one does not
measure a model — it measures its own parser, and reports the result as if it
were the model's.

**The repo demonstrates this rather than asserting it.** Every evaluation run
reports `micro_f1` and `micro_f1_raw` side by side, plus `normalization_gain`
as its own tracked metric, so the claim is reproduced on every run instead of
resting on a number in a README.

### The negative result

Few-shot scored **worse** than zero-shot: 0.74 against 0.93. Likely causes are
example selection skewing the label prior, and a longer prompt diluting
instruction adherence. It is recorded rather than deleted, because a
reproducible negative result is worth more than a quietly dropped experiment.

Caveat: the evaluation sets were small (30 and 50 rows). Treat differences as
directional.

---

## How it fits with my other projects

Scoring, aggregation across seeds, the report format and the regression gate all
come from **[eval-harness](https://github.com/mukundjaiswal/eval-harness)**, a
separate package shared with my other two projects. The harness knows nothing
about complaints or F1; it takes a callable, a dataset and metric names.

What belongs to *this* project is one file — `src/complaint_triage/eval_metrics.py`
— declaring what this task is judged on and which direction is an improvement.

| Project | What it demonstrates |
|---|---|
| [eval-harness](https://github.com/mukundjaiswal/eval-harness) | The shared evaluation layer |
| **this repo** | Prompting only. Where measured quality actually comes from |
| [agentic-rag-assistant-nutrition](https://github.com/mukundjaiswal/agentic-rag-assistant-nutrition) | Agentic retrieval with LLM-as-judge quality gates |

Worth noting: `micro_f1` cannot be expressed as an average of per-case numbers —
it is computed over the whole prediction set at once. The harness supports that
because a metric is any `Sequence[Observation] -> float`, not only a field
average. That generality is what makes the split real rather than cosmetic.

---

## Install

```bash
git clone https://github.com/mukundjaiswal/financial-complaint-triage.git
cd financial-complaint-triage

python -m venv .venv && source .venv/bin/activate
make install-dev          # installs the shared harness first
```

Running the tests needs neither a model file nor a GPU — the whole pipeline is
exercised against a scripted fake:

```bash
pytest
```

To run against a real model, add the optional extra and point `.env` at a local
GGUF instruction model:

```bash
pip install -e ".[model]"
cp .env.example .env
complaint-triage classify "my credit card was charged twice for one purchase"
```

## Evaluate

```bash
# baseline
complaint-triage evaluate --dataset evaluation/datasets/v1.jsonl \
  --out evaluation/baselines/zero-shot-v1.json

# later: compare across seeds, exit non-zero on a regression
complaint-triage evaluate --dataset evaluation/datasets/v1.jsonl \
  --baseline evaluation/baselines/zero-shot-v1.json --seeds 0,1,2
```

Metrics this project registers:

| Metric | Direction | What it tells you |
|---|---|---|
| `micro_f1` | higher | Accuracy on parsed predictions |
| `micro_f1_raw` | higher | The control — the same model with nothing reading its output |
| `normalization_gain` | higher | What the parsing layer is worth, tracked directly so erosion is visible |
| `unparseable_rate` | lower | **Harness health.** Rising means the model changed its output shape and the parser has not caught up |
| `summary_length_ratio` | lower | Guard against a model that returns its input and scores well for it |
| `p50` / `p95` latency | lower | From the harness |

`unparseable_rate` is the one to watch. A wrong answer is a model problem; an
unreadable one is a harness problem, and they need different fixes — so the
pipeline never collapses them into "incorrect".

---

## Layout

```
src/complaint_triage/
  categories.py     the label set and the aliases the model actually emits
  postprocess.py    output normalization   <- the 26-point file
  prompts.py        zero-shot and few-shot templates
  tasks.py          Classifier and Summarizer; both report raw AND parsed
  eval_metrics.py   what this task is judged on, registered with eval-harness
  interfaces.py     one Protocol, so the pipeline is testable offline
  llm/              GGUF adapter + a scripted fake
  settings.py       typed, validated config; secrets as SecretStr
  cli.py            classify / summarize / evaluate
tests/              offline; no weights, no GPU, no network
```

### Why a local quantized model

The task is high-volume routing. A hosted frontier model priced per token is
the wrong tool for a decision this narrow, and the measured result says the
capability was never the bottleneck — the scoring layer was.

---

## What this does not do

- **No committed dataset**, so the CI gate is scaffolded but not switched on.
  The [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
  is public-domain and maps onto this schema directly; that is the intended route.
- **Small evaluation sets** (30 and 50 rows). Differences are directional, not
  significant.
- **No per-class breakdown.** Micro-F1 hides which category regressed. The
  dataset format already carries arbitrary metadata for slice labels; nothing
  groups by it yet.
- **Summarization is under-evaluated.** `summary_length_ratio` is a guard, not a
  quality score. Reference-based scoring (BERTScore, ROUGE) is available behind
  the `scoring` extra but not wired into the default metric set.
- **No cost accounting.** Latency is measured; tokens are not.

## Development

```bash
make install-dev
make check          # ruff, mypy --strict, pytest
```

## License

MIT. See [LICENSE](LICENSE).
