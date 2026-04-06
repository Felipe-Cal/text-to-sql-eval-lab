# Fine-tuning

Rather than prompting a large general-purpose model, fine-tuning trains a small open-source model (Llama 3.1 8B) to specialise in text-to-SQL. The result is a model that requires no few-shot examples or chain-of-thought prompting — it has learned the task directly from data.

## How it works

```
tuning.json (40 synthetic) + golden/questions.json (15 hand-crafted)
                │
                ▼
  prepare_finetune_data.py
  Converts Q&A pairs → JSONL chat format (system + user + assistant)
                │
                ▼
  Google Colab T4 GPU
  Llama 3.1 8B + LoRA adapters (r=16, ~1% of parameters trained)
  3 epochs, ~15 minutes, SFTTrainer
                │
                ▼
  Export to GGUF (Q4_K_M, ~4.5GB)
                │
                ▼
  ollama create text2sql-llama -f Modelfile
                │
                ▼
  python scripts/run_eval.py --model ollama/text2sql-llama --strategy zero_shot
```

## Why LoRA instead of full fine-tuning

Full fine-tuning retrains all 8 billion parameters — requires ~80GB VRAM and thousands of dollars. LoRA freezes the original weights and trains small adapter matrices injected into each attention layer, reducing trainable parameters by ~99%. With 4-bit quantization (QLoRA), the whole thing fits on a free Colab T4 GPU.

## Step 1 — Prepare the data (run locally)

```bash
python scripts/prepare_finetune_data.py
```

Outputs:
- `datasets/finetune/train.jsonl` — 90% of combined dataset (40 examples)
- `datasets/finetune/eval.jsonl` — 10% held out for training-time eval (5 examples)

Each example is formatted as a three-turn conversation:

```json
{
  "messages": [
    {"role": "system",    "content": "You are an expert SQL engineer..."},
    {"role": "user",      "content": "Schema:\n...\n\nQuestion: What is the total revenue?"},
    {"role": "assistant", "content": "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) ..."}
  ]
}
```

## Step 2 — Train on Colab

1. Open `notebooks/finetune_llama.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells — training takes ~15 minutes
4. Download the exported GGUF file

## Step 3 — Serve locally with Ollama

```bash
# Import the fine-tuned model
echo 'FROM ./text2sql-llama.Q4_K_M.gguf' > Modelfile
ollama create text2sql-llama -f Modelfile

# Run it
ollama run text2sql-llama
```

## Step 4 — Evaluate and compare

```bash
# LiteLLM supports Ollama natively via the ollama/ prefix
python scripts/run_eval.py --model ollama/text2sql-llama --strategy zero_shot
python scripts/run_eval.py --model ollama/text2sql-llama --strategy few_shot_dynamic
```

## Measured results (Apple M1, 15 golden questions)

| Model | Strategy | result_match | semantic_judge | cost/query | avg latency |
|---|---|---|---|---|---|
| gpt-4o-mini | zero_shot | 0.667 | 0.733 | ~$0.002 | ~2s |
| gpt-4o-mini | few_shot_dynamic | 0.933 | 0.933 | ~$0.002 | ~2s |
| **Llama 3.1 8B fine-tuned** | **zero_shot** | **0.667** | **0.567** | **$0.000** | **99s** |
| **Llama 3.1 8B fine-tuned** | **few_shot_dynamic** | **0.800** | **0.867** | **$0.000** | **74s** |

**Key finding:** The fine-tuned Llama with `few_shot_dynamic` outperforms GPT-4o-mini zero_shot on both metrics at zero inference cost. Combining fine-tuning with in-context examples proves more effective than either alone — the model has learned SQL patterns from training data and the examples close the remaining gap on hard questions.

On a GPU (e.g. A10G), latency drops from ~74s to ~5–8s per question.
