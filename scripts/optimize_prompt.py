import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

import dspy

# Ensure project root is on sys.path when run as:
#   python scripts/optimize_prompt.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.db import execute_query
from src.evals.scorers import _normalize_rows
from src.agent.schema_retriever import retrieve_schema

load_dotenv()

class TextToSQL(dspy.Signature):
    """Convert a natural language question about an enterprise database into a duckdb SQL query. Only output valid duckdb SQL, no markdown fences."""
    schema = dspy.InputField(desc="The database schema definition containing relevant tables.", format=str)
    question = dspy.InputField(desc="The natural language question to answer.", format=str)
    sql = dspy.OutputField(desc="The single DuckDB SQL query answering the question.", format=str)

class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought to force the model to reason before outputting SQL
        self.generate = dspy.ChainOfThought(TextToSQL)
        
    def forward(self, schema, question):
        return self.generate(schema=schema, question=question)

def load_dataset(path: str) -> list[dspy.Example]:
    with open(path) as f:
        data = json.load(f)
        
    examples = []
    print(f"Loading {len(data)} examples and fetching RAG schemas...")
    for q in data:
        schema, _ = retrieve_schema(q["question"], top_k=5)
        # Store expected_rows internally so the metric can evaluate it
        ex = dspy.Example(
            question=q["question"], 
            sql=q["sql"],
            schema=schema,
            expected_rows=q["expected_rows"]
        ).with_inputs("schema", "question")
        examples.append(ex)
    return examples

def db_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> int:
    """Metric: Executes predicted SQL and strictly compares output rows to expected rows."""
    try:
        # Strip potential markdown fences just in case 
        raw_sql = pred.sql.replace("```sql", "").replace("```", "").strip()
        actual = execute_query(raw_sql)
        norm_actual = _normalize_rows(actual)
        norm_expected = _normalize_rows(example.expected_rows)
        return int(str(norm_actual) == str(norm_expected))
    except Exception:
        # Fails duckdb execution or formatting
        return 0

def main():
    # 1. Setup LM
    model_id = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
    lm = dspy.LM(model_id, api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)
    
    # 2. Load the Tuning Dataset generated in Phase 2
    tuning_set = load_dataset("datasets/synthetic/tuning.json")
    
    from dspy.teleprompt import BootstrapFewShot
    
    print(f"\nStarting DSPy optimization with {len(tuning_set)} synthetic examples...")
    teleprompter = BootstrapFewShot(
        metric=db_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_errors=3
    )
    
    # 3. Compile the program
    optimizer = teleprompter.compile(SQLGenerator(), trainset=tuning_set)
    
    # 4. Save the compiled prompts
    out_path = Path(__file__).parent.parent / "datasets" / "dspy_optimized_prompt.json"
    optimizer.save(str(out_path))
    print(f"\nOptimization complete! Compiled module saved to {out_path}")

if __name__ == "__main__":
    main()
