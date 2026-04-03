import os
from pathlib import Path
import dspy

class TextToSQL(dspy.Signature):
    """Convert a natural language question about an enterprise database into a duckdb SQL query. Only output valid duckdb SQL, no markdown fences."""
    schema = dspy.InputField(desc="The database schema definition containing relevant tables.", format=str)
    question = dspy.InputField(desc="The natural language question to answer.", format=str)
    sql = dspy.OutputField(desc="The single DuckDB SQL query answering the question.", format=str)

class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(TextToSQL)
        
    def forward(self, schema, question):
        return self.generate(schema=schema, question=question)

_dspy_lm = None
_dspy_agent = None

def get_dspy_agent(model_id: str):
    """Initialize DSPy LM and load the optimized prompt weights."""
    global _dspy_lm, _dspy_agent
    
    if _dspy_lm is None:
        _dspy_lm = dspy.LM(model_id, api_key=os.getenv("OPENAI_API_KEY"), cache=False)
        dspy.configure(lm=_dspy_lm)
        
    if _dspy_agent is None:
        _dspy_agent = SQLGenerator()
        weight_path = Path(__file__).parent.parent.parent / "datasets" / "dspy_optimized_prompt.json"
        if weight_path.exists():
            _dspy_agent.load(str(weight_path))
        else:
            print(f"Warning: DSPy weights not found at {weight_path}. Running unoptimized.")
            
    return _dspy_agent, _dspy_lm
