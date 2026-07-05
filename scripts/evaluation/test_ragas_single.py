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

import sys
import types
try:
    from langchain_google_vertexai import ChatVertexAI   # type: ignore
except ImportError:
    class ChatVertexAI:
        pass
# Register dummy fallback module to satisfy old Ragas imports
vertexai_module = types.ModuleType("vertexai")
vertexai_module.ChatVertexAI = ChatVertexAI              # type: ignore
sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module

from typing import Optional, Dict, Any, List, Sequence
from config.logging_config import setup_logging, logger
from config.settings import settings
from scripts.evaluation.shared.generation_bioasq_common import (
    initialize_ragas_evaluator,
    get_ragas_metric_keys,
)


def evaluate_ragas_question(
    evaluator: Optional[Dict[str, Any]],
    question: str,
    generated_answer: str,
    contexts: List[str],
    references: Sequence[str],
) -> Dict[str, float]:
    """Evaluate one question with RAGAS and average across references using split groups."""
    if evaluator is None:
        return {}

    cleaned_refs = [ref.strip() for ref in references if isinstance(ref, str) and ref.strip()]
    if not cleaned_refs:
        return {}

    indep_metric_names = ["faithfulness", "answer_relevancy"]
    dep_metric_names = ["context_precision", "context_recall", "answer_correctness"]

    indep_metrics = [m for m in evaluator["metrics"] if m.name in indep_metric_names]
    dep_metrics = [m for m in evaluator["metrics"] if m.name in dep_metric_names]

    from ragas.run_config import RunConfig
    run_config = RunConfig(
        max_workers=1,
        max_retries=10,
        max_wait=60,
        timeout=1800,
    )

    # 1. Evaluate independent metrics (Group 1) - once
    first_ref = cleaned_refs[0] if cleaned_refs else ""
    dataset_indep = evaluator["dataset_cls"].from_dict(
        {
            "question": [question],
            "answer": [generated_answer],
            "contexts": [contexts],
            "ground_truth": [first_ref],
        }
    )

    indep_scores = {}
    if indep_metrics:
        try:
            result_indep = evaluator["evaluate_fn"](
                dataset_indep,
                metrics=indep_metrics,
                llm=evaluator["llm"],
                embeddings=evaluator["embeddings"],
                run_config=run_config,
                raise_exceptions=False,
            )
            row = result_indep.to_pandas().iloc[0].to_dict()
            for key in indep_metric_names:
                val = row.get(key)
                if isinstance(val, (int, float)):
                    indep_scores[key] = float(val)
        except Exception as exc:
            logger.error(f"Single question independent RAGAS evaluation failed: {exc}")

    # 2. Evaluate dependent metrics (Group 2) - once per reference
    dep_scores_list = []
    if dep_metrics:
        for reference in cleaned_refs:
            dataset_dep = evaluator["dataset_cls"].from_dict(
                {
                    "question": [question],
                    "answer": [generated_answer],
                    "contexts": [contexts],
                    "ground_truth": [reference],
                }
            )
            try:
                result_dep = evaluator["evaluate_fn"](
                    dataset_dep,
                    metrics=dep_metrics,
                    llm=evaluator["llm"],
                    embeddings=evaluator["embeddings"],
                    run_config=run_config,
                    raise_exceptions=False,
                )
                row = result_dep.to_pandas().iloc[0].to_dict()
                scores = {}
                for key in dep_metric_names:
                    val = row.get(key)
                    if isinstance(val, (int, float)):
                        scores[key] = float(val)
                dep_scores_list.append(scores)
            except Exception as exc:
                logger.error(f"Single question dependent RAGAS evaluation failed for reference: {exc}")

    # Average dependent scores
    dep_avg_scores = {}
    for key in dep_metric_names:
        vals = [s[key] for s in dep_scores_list if key in s]
        if vals:
            dep_avg_scores[key] = sum(vals) / len(vals)

    # Combine and return
    combined = {}
    for key in get_ragas_metric_keys():
        if key in indep_scores:
            combined[key] = round(indep_scores[key], 4)
        elif key in dep_avg_scores:
            combined[key] = round(dep_avg_scores[key], 4)

    return combined


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
