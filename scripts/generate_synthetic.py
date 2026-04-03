import json
import random
import os
from pathlib import Path
import litellm
from dotenv import load_dotenv

from src.utils.db import get_schema_string, execute_query
from src.evals.scorers import _normalize_rows

load_dotenv()

PROMPT = """
You are an expert DuckDB SQL engineer. Given the following schema, generate 10 novel, advanced, and varied SQL questions.
Make sure they are "medium" or "hard" difficulty (using multiple JOINs, aggregations, subqueries, HAVING, time series, etc.).
Ensure that your queries will actually return results and not be empty datasets given a standard realistic seeded database.

Schema:
{schema}

Return EXACTLY a JSON object with a "questions" key containing a list. No markdown fences, no other text.
Format:
{{
  "questions": [
    {{
      "question": "What is the total revenue per category for customers in the USA?",
      "sql": "SELECT p.category, SUM(oi.quantity * oi.unit_price) FROM ...",
      "difficulty": "medium",
      "notes": "JOIN across 4 tables with aggregation"
    }}
  ]
}}
"""

def generate_batch(model: str, schema: str) -> list[dict]:
    print(f"Asking {model} for 10 questions...")
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(schema=schema)}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        return data.get("questions", [])
    except Exception as e:
        print(f"Failed to parse LLM output: {e}\n{raw}")
        return []

def main():
    model = os.getenv("DEFAULT_MODEL", "openai/gpt-4o")
    schema = get_schema_string()
    
    target_count = 50
    valid_samples = []
    
    attempts = 0
    max_attempts = 15
    
    print("Starting synthetic data generation Data Flywheel...")
    print(f"Target: {target_count} valid queries")
    
    while len(valid_samples) < target_count and attempts < max_attempts:
        attempts += 1
        batch = generate_batch(model, schema)
        
        for item in batch:
            if len(valid_samples) >= target_count:
                break
                
            sql = item.get("sql", "")
            if not sql:
                continue
                
            # Try to execute
            try:
                rows = execute_query(sql)
                if len(rows) == 0:
                    # Skip empty result sets - they make poor eval targets
                    continue
                    
                # Normalize so it can be JSON serialized (floats, ISO dates)
                normalized_rows = _normalize_rows(rows)
                
                # Turn normalized tuples into lists (JSON arrays)
                item["expected_rows"] = [list(r) for r in normalized_rows]
                item["id"] = f"sq{len(valid_samples) + 1:03d}"
                
                valid_samples.append(item)
                print(f"[{len(valid_samples)}/{target_count}] Valid: {item['question']}")
                
            except Exception as e:
                # Discard invalid SQL gracefully
                pass

    print(f"\nGenerated {len(valid_samples)} valid text-to-SQL pairs.")
    
    # Shuffle and split
    random.shuffle(valid_samples)
    split_idx = int(len(valid_samples) * 0.8) # 80% tuning, 20% holdout
    
    tuning_set = valid_samples[:split_idx]
    holdout_set = valid_samples[split_idx:]
    
    out_dir = Path(__file__).parent.parent / "datasets" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "tuning.json", "w") as f:
        json.dump(tuning_set, f, indent=2)
        
    with open(out_dir / "holdout_test.json", "w") as f:
        json.dump(holdout_set, f, indent=2)
        
    print(f"Saved {len(tuning_set)} samples to datasets/synthetic/tuning.json")
    print(f"Saved {len(holdout_set)} samples to datasets/synthetic/holdout_test.json")

if __name__ == "__main__":
    main()
