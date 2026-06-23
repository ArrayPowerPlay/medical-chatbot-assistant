"""
MedAESQA retrieval tuning script (One-Factor-At-A-Time).

Evaluates text retrieval configurations by varying one parameter at a time
from a strong baseline on the first 15 questions of the MedAESQA test set.
Parameter k_rrf is kept constant at 60.

Usage:
    python scripts/evaluation/medaesqa/tuning/tune_retrieval.py --limit 15
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import logger, setup_logging
from config.settings import settings
from scripts.evaluation.shared import retrieval_common as er
from src.embeddings.medcpt_embedder import MedCPTEmbedder
from src.query.query_analyzer import QueryAnalyzer
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.rrf import RRFManager
from src.storage.parent_store import ParentStore
from src.storage.weaviate_client import WeaviateChildStore

TEST_PATH = settings.DATA_PATH / "test" / "test_medaesqa.jsonl"
OUTPUT_DIR = settings.EVAL_RESULTS_PATH / "medaesqa" / "retrieval"
K_VALUES = settings.K_VALUES

# Define the baseline parameter set
BASELINE: Dict[str, int] = {
    "vector_top_k": 40,
    "keyword_top_k": 80,
    "child_fetch_limit": 120,
    "top_k_rrf": 80,
    "k_rrf": 60,  
    "rerank_text_top_m": 20,
}

# Parameters to vary one-factor-at-a-time (OFAT)
CANDIDATE_VARY: Dict[str, List[int]] = {
    "vector_top_k": [60, 80],
    "keyword_top_k": [60, 100],
    "child_fetch_limit": [150],
    "top_k_rrf": [60, 100],
    "rerank_text_top_m": [10, 30],
}


def load_questions(limit: int) -> List[Dict[str, Any]]:
    """Load the first N test questions."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    questions: List[Dict[str, Any]] = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    return questions[:limit]


def apply_retrieval_settings(params: Dict[str, int]) -> None:
    """Mutate settings for this tuning run."""
    settings.VECTOR_TOP_K = params["vector_top_k"]
    settings.KEYWORD_TOP_K = params["keyword_top_k"]
    settings.CHILD_FETCH_LIMIT = params["child_fetch_limit"]
    settings.TOP_K_RRF = params["top_k_rrf"]
    settings.K_RRF = params["k_rrf"]
    settings.RERANK_TEXT_TOP_M = params["rerank_text_top_m"]


def evaluate_config(
    questions: List[Dict[str, Any]],
    query_analyzer: QueryAnalyzer,
    query_embedder: MedCPTEmbedder,
    weaviate_store: WeaviateChildStore,
    parent_store: ParentStore,
    rrf_manager: RRFManager,
    cross_encoder: CrossEncoderReranker,
    debug_label: str,
) -> Dict[str, float]:
    """Evaluate questions on a specific configuration and return mean metrics."""
    all_metrics: List[Dict[str, float]] = []

    for q in questions:
        q_id = q["id"]
        body = q["body"]
        gold_pmids: Set[str] = set(q.get("relevant_pmid", []))

        if not gold_pmids:
            continue

        try:
            ranked_text, _ = er.run_retrieval_pipeline(
                query=body,
                query_analyzer=query_analyzer,
                query_embedder=query_embedder,
                weaviate_store=weaviate_store,
                parent_store=parent_store,
                rrf_manager=rrf_manager,
                cross_encoder=cross_encoder,
                debug_label=f"{debug_label}-{q_id}",
            )

            # Deduplicate PMIDs while preserving rank order
            seen: Set[str] = set()
            retrieved_pmids: List[str] = []
            for item in ranked_text:
                pmid = item.get("pmid", "")
                if pmid and pmid not in seen:
                    seen.add(pmid)
                    retrieved_pmids.append(pmid)

            # Compute Precision, Recall, F1 at K
            q_metrics: Dict[str, float] = {}
            for k in K_VALUES:
                p = er.precision_at_k(retrieved_pmids, gold_pmids, k)
                r = er.recall_at_k(retrieved_pmids, gold_pmids, k)
                f = er.f1_score(p, r)
                q_metrics[f"P@{k}"] = p
                q_metrics[f"R@{k}"] = r
                q_metrics[f"F1@{k}"] = f

            all_metrics.append(q_metrics)

        except Exception:
            logger.exception(f"Failed to retrieve for question {q_id}")

    # Compute mean metrics
    mean_metrics: Dict[str, float] = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            mean_metrics[key] = round(sum(m[key] for m in all_metrics) / len(all_metrics), 4)

    return mean_metrics


def run_tuning(limit: int) -> None:
    """Run OFAT tuning and print results."""
    logger.info(f"Starting MedAESQA retrieval OFAT tuning on first {limit} questions...")
    questions = load_questions(limit)

    # Save original settings to restore later
    original_settings = {
        "vector_top_k": settings.VECTOR_TOP_K,
        "keyword_top_k": settings.KEYWORD_TOP_K,
        "child_fetch_limit": settings.CHILD_FETCH_LIMIT,
        "top_k_rrf": settings.TOP_K_RRF,
        "k_rrf": settings.K_RRF,
        "rerank_text_top_m": settings.RERANK_TEXT_TOP_M,
    }

    query_analyzer = QueryAnalyzer()
    query_analyzer.temperature = 0.0
    query_embedder = MedCPTEmbedder(mode="query")
    weaviate_store = WeaviateChildStore()
    parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)
    cross_encoder = CrossEncoderReranker()

    # Construct the list of configurations to test
    configs: List[Dict[str, Any]] = []
    
    # 1. Baseline
    configs.append({
        "name": "Baseline",
        "params": BASELINE.copy()
    })
    
    # 2. OFAT Variations
    for param, values in CANDIDATE_VARY.items():
        for val in values:
            cfg_params = BASELINE.copy()
            cfg_params[param] = val
            configs.append({
                "name": f"{param}={val}",
                "params": cfg_params
            })

    results: List[Dict[str, Any]] = []

    try:
        for idx, cfg in enumerate(configs, 1):
            name = cfg["name"]
            params = cfg["params"]

            logger.info(f"\n--- Run {idx}/{len(configs)}: {name} ---")
            apply_retrieval_settings(params)
            
            # Re-create RRFManager to ensure the current K_RRF is respected
            rrf_manager = RRFManager(k=settings.K_RRF)

            metrics = evaluate_config(
                questions=questions,
                query_analyzer=query_analyzer,
                query_embedder=query_embedder,
                weaviate_store=weaviate_store,
                parent_store=parent_store,
                rrf_manager=rrf_manager,
                cross_encoder=cross_encoder,
                debug_label=f"Tune-{idx}",
            )

            results.append({
                "config_name": name,
                "params": params,
                "metrics": metrics
            })

        # Print comparative results table
        print("\n" + "=" * 100)
        print(f"MEDAESQA RETRIEVAL TUNING COMPARISON (OFAT, Limit={limit} questions)")
        print("=" * 100)
        print(f"{'Configuration':<25} | {'P@5':<8} {'R@5':<8} {'F1@5':<8} | {'P@10':<8} {'R@10':<8} {'F1@10':<8} | {'P@20':<8} {'R@20':<8} {'F1@20':<8}")
        print("-" * 100)
        for res in results:
            name = res["config_name"]
            m = res["metrics"]
            print(
                f"{name:<25} | "
                f"{m.get('P@5', 0.0):.4f} "
                f"{m.get('R@5', 0.0):.4f} "
                f"{m.get('F1@5', 0.0):.4f} | "
                f"{m.get('P@10', 0.0):.4f} "
                f"{m.get('R@10', 0.0):.4f} "
                f"{m.get('F1@10', 0.0):.4f} | "
                f"{m.get('P@20', 0.0):.4f} "
                f"{m.get('R@20', 0.0):.4f} "
                f"{m.get('F1@20', 0.0):.4f}"
            )
        print("=" * 100)

        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"tune_summary_ofat_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "limit": limit,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Tuning summary saved to {output_file}")

    finally:
        # Restore original settings
        apply_retrieval_settings(original_settings)

        query_analyzer.close()
        query_embedder.close()
        if hasattr(cross_encoder, "close"):
            cross_encoder.close()
        weaviate_store.close()
        parent_store.close()


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Tune MedAESQA retrieval parameters using OFAT.")
    parser.add_argument("--limit", type=int, default=15, help="Number of questions to evaluate (default: 15).")
    args = parser.parse_args()

    run_tuning(args.limit)
