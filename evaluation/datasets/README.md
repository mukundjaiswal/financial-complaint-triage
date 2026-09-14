# Evaluation datasets

Datasets are **not** committed. The complaint corpus used to develop this was
provided for a course and is not mine to redistribute.

The task expects JSON Lines with the keys this project documents — the shared
harness is told to read them via `load_cases(..., input_key="narrative",
reference_key="product")`, rather than the datasets being rewritten to suit it:

```json
{"narrative": "free text complaint", "product": "credit_card", "summary": "reference summary"}
```

| Field | Required | Purpose |
|---|---|---|
| `narrative` | yes | The complaint text |
| `product` | for classification | Gold label |
| `summary` | for summarization | Reference summary |
| any other key | no | Preserved as metadata; use it for slice labels |

`product` must be one of: `credit_card`, `retail_banking`, `credit_reporting`,
`mortgages_and_loans`, `debt_collection`.

## A public substitute

The **CFPB Consumer Complaint Database** is US public-domain data with free-text
narratives and product labels, and maps onto this schema directly. It is the
intended route to a committed `v1.jsonl` and a CI gate that actually runs.

## Version the filename

Name it `v1.jsonl`, `v2.jsonl`, and never edit a version in place. Every report
records the dataset it ran against, and two reports compare only if that string
matches. Editing `v1.jsonl` silently invalidates every baseline committed
against it.
