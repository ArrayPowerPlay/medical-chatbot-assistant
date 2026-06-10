"""
Simple script to test if the RAGAS evaluation framework is working.
This script runs RAGAS on a single mock medical question-answer pair.
It does not require Weaviate, Neo4j, or Modal to be active, as it uses static inputs.
It only requires a valid OPENAI_API_KEY in the .env file.

Usage:
    python scripts/evaluation/test_ragas_single.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import setup_logging
from config.settings import settings
from scripts.evaluation.shared.generation_bioasq_common import (
    initialize_ragas_evaluator,
    evaluate_ragas_question,
)


def run_test():
    print("=" * 60)
    print("RAGAS CONNECTION TEST")
    print("=" * 60)

    # 1. Check OpenAI API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[-] ERROR: OPENAI_API_KEY is not set in the environment or .env file.")
        print("    Please check your .env file or export the key.")
        return False

    # Mask key for safety
    masked_key = api_key[:8] + "..." + api_key[-8:] if len(api_key) > 16 else "invalid"
    print(f"[+] Found OPENAI_API_KEY: {masked_key}")
    print(f"[+] RAGAS LLM Model: {settings.RAGAS_EVALUATOR_LLM_MODEL}")
    print(f"[+] RAGAS Embedding Model: {settings.RAGAS_EVALUATOR_EMBEDDING_MODEL}")
    print("-" * 60)

    # 2. Initialize RAGAS Evaluator
    print("[*] Initializing RAGAS evaluator components...")
    try:
        evaluator = initialize_ragas_evaluator(enabled=True)
    except Exception as e:
        print(f"[-] ERROR: Failed to import or initialize RAGAS dependencies: {e}")
        return False

    if evaluator is None:
        print("[-] ERROR: initialize_ragas_evaluator returned None.")
        print("    Please check the logs or ensure dependency packages are installed:")
        print("    pip install ragas langchain-openai datasets")
        return False

    print("[+] RAGAS evaluator initialized successfully!")
    print("-" * 60)

    # 3. Define a mock medical question-answer pair for evaluation
    question = "Is Metformin used to treat Type 2 Diabetes?"
    generated_answer = (
        "Yes, Metformin is a first-line medication used to treat Type 2 Diabetes. "
        "It works by improving insulin sensitivity and lowering glucose production in the liver."
    )
    contexts = [
        "Metformin is an oral diabetes medicine that helps control blood sugar levels. "
        "It is indicated for the treatment of patients with type 2 diabetes mellitus."
    ]
    ground_truth = [
        "Yes, Metformin is widely prescribed as an initial pharmacological agent for managing Type 2 Diabetes."
    ]

    print("Evaluating test case:")
    print(f"  Question:      {question}")
    print(f"  Answer (Gen):  {generated_answer}")
    print(f"  Contexts:      {contexts}")
    print(f"  Ground Truth:  {ground_truth}")
    print("-" * 60)

    # 4. Run evaluation
    print("[*] Sending request to OpenAI via RAGAS (this may take a few seconds)...")
    try:
        scores = evaluate_ragas_question(
            evaluator=evaluator,
            question=question,
            generated_answer=generated_answer,
            contexts=contexts,
            references=ground_truth,
        )
    except Exception as e:
        print(f"[-] ERROR during RAGAS evaluation: {e}")
        return False

    # 5. Output results
    if not scores:
        print("[-] ERROR: RAGAS evaluation returned empty scores.")
        print("    This usually indicates an API key issue or rate limit.")
        return False

    print("[+] RAGAS evaluation completed successfully! Results:")
    for metric, score in scores.items():
        print(f"  - {metric:20}: {score:.4f}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    setup_logging()
    success = run_test()
    sys.exit(0 if success else 1)
