import os
import sys
import json
import groq

# Add root project to sys.path so we can import src
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.query.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()

history = [
    {"role": "user", "content": "How do I manage my asthma?"},
    {"role": "assistant", "content": "Asthma can be managed using inhalers like Albuterol and avoiding triggers."}
]
query = "What are the common symptoms of psoriasis?"

print("--- RUNNING TEST ---")
result = analyzer.analyze(query, history=history)
print(json.dumps(result, indent=2, ensure_ascii=False))
