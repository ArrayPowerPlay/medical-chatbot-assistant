"""
Experiment: Child-to-Parent Ratio Analysis

Runs a small set of questions from the BioASQ validation set through the
retrieval pipeline (Vector + BM25) and measures how many unique parent chunks
result from a given number of fetched child chunks.

Purpose: Determine the optimal `child_fetch_limit` for a desired number of
unique parent results (eval_top_k). The ratio guides the setting:
    child_fetch_limit = ratio * eval_top_k

Usage:
    python scripts/experiment_child_ratio.py --num_questions 20 --child_limits 40 60 80 100
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Configure project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from config.logging_config import logger
from src.embeddings.medcpt_embedder import MedCPTEmbedder
from src.storage.weaviate_client import WeaviateChildStore
from src.storage.parent_store import ParentStore


def count_unique_parents(child_results: list) -> int:
    """Count the number of unique parent IDs in a list of child results.

    Args:
        child_results: List of child chunk dicts, each with a 'parent_id' key.

    Returns:
        Number of unique parent IDs found.
    """
    return len({c["parent_id"] for c in child_results})


def run_experiment(
    num_questions: int,
    child_limits: list,
    val_path: Path
) -> None:
    """Run the child-to-parent ratio experiment.

    For each question, fetches child chunks at various limits via both
    Vector Search and BM25, then counts how many unique parents each
    limit produces.

    Args:
        num_questions: Number of questions to sample from the validation set.
        child_limits: List of child_fetch_limit values to test (e.g. [40, 60, 80]).
        val_path: Path to the validation JSONL file.
    """
    # Load validation questions
    questions = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    questions = questions[:num_questions]
    logger.info(f"Loaded {len(questions)} questions for experiment.")

    # Initialize components
    query_embedder = MedCPTEmbedder(mode="query")
    weaviate_store = WeaviateChildStore()
    parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)

    # Results storage: {child_limit: {"vector": [counts], "bm25": [counts], "union": [counts]}}
    results = {cl: {"vector": [], "bm25": [], "union": []} for cl in child_limits}

    try:
        for i, q in enumerate(questions):
            body = q["body"]
            logger.info(f"[{i+1}/{len(questions)}] Processing: {body[:80]}...")

            # Encode query
            query_vector = query_embedder.embed_texts(body)[0]

            for cl in child_limits:
                # Vector search
                vec_children = weaviate_store.vector_search(
                    query_vector=query_vector, limit=cl
                )
                vec_parents = count_unique_parents(vec_children)

                # BM25 search
                bm25_children = weaviate_store.bm25_search(
                    query_text=body, limit=cl
                )
                bm25_parents = count_unique_parents(bm25_children)

                # Union (simulating RRF input)
                all_parent_ids = {c["parent_id"] for c in vec_children}
                all_parent_ids.update(c["parent_id"] for c in bm25_children)
                union_parents = len(all_parent_ids)

                results[cl]["vector"].append(vec_parents)
                results[cl]["bm25"].append(bm25_parents)
                results[cl]["union"].append(union_parents)

        # Print results table
        print("\n" + "=" * 75)
        print("CHILD-TO-PARENT RATIO EXPERIMENT RESULTS")
        print(f"Questions tested: {len(questions)}")
        print("=" * 75)
        print(f"{'child_limit':>12} | {'Stream':>8} | {'Min':>5} | {'Avg':>7} | {'Max':>5} | {'Ratio':>7}")
        print("-" * 75)

        for cl in child_limits:
            for stream in ["vector", "bm25", "union"]:
                counts = results[cl][stream]
                avg = sum(counts) / len(counts)
                ratio = cl / avg if avg > 0 else float("inf")
                print(
                    f"{cl:>12} | {stream:>8} | {min(counts):>5} | {avg:>7.1f} | "
                    f"{max(counts):>5} | {ratio:>7.1f}"
                )
            print("-" * 75)

        # Recommendation
        print("\nRECOMMENDATION:")
        print("For eval_top_k=20, choose child_fetch_limit where 'union avg' >= 30")
        print("(provides headroom for RRF deduplication and Cross-Encoder filtering)")

    finally:
        weaviate_store.close()
        parent_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment: measure child-to-parent ratio for retrieval evaluation."
    )
    parser.add_argument(
        "--num_questions", type=int, default=20,
        help="Number of validation questions to test (default: 20)."
    )
    parser.add_argument(
        "--child_limits", type=int, nargs="+", default=[40, 60, 80, 100],
        help="List of child_fetch_limit values to test (default: 40 60 80 100)."
    )
    args = parser.parse_args()

    val_path = settings.DATA_PATH / "val" / "val_bioasq.jsonl"
    if not val_path.exists():
        print(f"Error: {val_path} not found. Run preprocess_bioasq_taskB.py first.")
        sys.exit(1)

    run_experiment(
        num_questions=args.num_questions,
        child_limits=args.child_limits,
        val_path=val_path,
    )
