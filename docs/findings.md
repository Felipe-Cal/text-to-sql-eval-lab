# Experimental findings

## GPT-4o-mini strategy sweep (15 questions)

| Strategy | result_match | semantic_judge | time |
|---|---|---|---|
| `zero_shot` | 0.667 | 0.733 | 0:48 |
| `few_shot_static` | 0.767 | 0.767 | 0:48 |
| `few_shot_dynamic` | **0.933** | **0.933** | 0:52 |
| `chain_of_thought` | 0.667 | 0.800 | 1:16 |

## Fine-tuned Llama 3.1 8B vs GPT-4o-mini (15 questions)

| Model | Strategy | result_match | semantic_judge | avg_cost | avg_latency |
|---|---|---|---|---|---|
| gpt-4o-mini | zero_shot | 0.667 | 0.733 | ~$0.002 | ~2s |
| gpt-4o-mini | few_shot_dynamic | 0.933 | 0.933 | ~$0.002 | ~2s |
| Llama 3.1 8B (fine-tuned) | zero_shot | 0.667 | 0.567 | **$0.000** | 99s |
| Llama 3.1 8B (fine-tuned) | few_shot_dynamic | **0.800** | **0.867** | **$0.000** | 74s |

## Key findings

- **`few_shot_dynamic` is the recommended strategy for cloud models** — +40% result_match over zero_shot for gpt-4o-mini, only +4s latency, and the only strategy where result_match and semantic_judge are fully aligned.

- **Fine-tuned Llama + `few_shot_dynamic` beats GPT-4o-mini zero_shot at zero cost** — result_match 0.800 vs 0.667 (+20%), semantic_judge 0.867 vs 0.733 (+18%). Domain-specific fine-tuning + in-context examples compounds: the model has learned SQL patterns from training data and the examples close the remaining gap on hard questions.

- **Fine-tuning alone (zero_shot) matches but doesn't exceed GPT-4o-mini** — result_match 0.667 = 0.667, but semantic_judge is weaker (0.567 vs 0.733). The model produces structurally correct SQL but makes semantic errors on harder multi-table questions without examples.

- **`few_shot_static` is inconsistent** — it helped `claude-haiku` significantly but *hurt* `gpt-4o-mini`'s semantic score vs zero_shot. Static examples can bias smaller models toward a specific SQL style, trading semantic correctness for syntactic similarity.

- **Chain-of-thought underperformed and is not recommended for this model/task** — −10% result_match vs zero_shot, +58% latency. Root cause: CoT causes *column selection drift* — 4 of 5 failures used `SELECT *` or returned extra columns. CoT benefits larger models more; `gpt-4o-mini` lacks the reasoning capacity to reliably apply it.

- **LLM judge calibration matters** — the initial `semantic_judge` produced false partial verdicts on `q02`, `q10`, and `q13` due to Python datetime reprs being passed as-is to the judge. Fixed by pre-normalizing result rows (ISO dates, rounded floats) and adding explicit grounding rules to the judge prompt.

> **Tip:** cases where `result_match = 0` but `semantic_judge = 1` are false negatives in the deterministic scorer — the model was semantically correct but returned rows in a different format. Cases where `result_match > semantic_judge` indicate potential judge calibration issues worth investigating.

## Exploring logs

```bash
python -m inspect_ai view start --log-dir ./logs
```

If `inspect view` alone shows empty rows or SSL errors, your shell may be picking a different `inspect` binary — use the command above or pass `--log-dir ./logs` explicitly.

Open the URL printed in the terminal (default `http://127.0.0.1:7575`).
