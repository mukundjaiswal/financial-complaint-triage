# Baselines

Commit a report here once a configuration is accepted as the reference point:

```bash
complaint-triage evaluate --dataset evaluation/datasets/v1.jsonl \
  --out evaluation/baselines/zero-shot-v1.json
```

Later runs become a diff against a committed number rather than an argument
from memory:

```bash
complaint-triage evaluate --dataset evaluation/datasets/v1.jsonl \
  --baseline evaluation/baselines/zero-shot-v1.json --tolerance 0.02
```

Non-zero exit on a regression is what lets CI gate on it. Direction is per
metric and travels on the report: higher `micro_f1` is better, higher
`unparseable_rate` is worse.

Keep separate baselines per configuration — `zero-shot-v1.json`,
`few-shot-10-v1.json`. Comparing a few-shot run against a zero-shot baseline
measures the prompt change, not a regression.

Re-baseline deliberately, in its own commit, with the reason in the message. A
baseline quietly refreshed alongside a behaviour change measures nothing.
