"""
MedAESQA test retrieval evaluation entrypoint.

Evaluates text retrieval quality using PMID-level Precision, Recall, and F1 
against the expert-curated relevant_pmid field in the MedAESQA test set.

Notes:
  - KG paths are excluded: they carry no PMID and therefore cannot contribute
    to PMID-level P/R/F1 metrics.
  - Only P/R/F1@K are reported (K = 5/10/20)

Outputs:
    results/test_results/medaesqa/retrieval/detail.jsonl
    results/test_results/medaesqa/retrieval/summary.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import logger, setup_logging
from config.settings import settings
from scripts.evaluation.shared.retrieval_common import (
    build_arg_parser,
    f1_score,
    precision_at_k,
    recall_at_k,
    run_retrieval_pipeline,
)
from src.embeddings.medcpt_embedder import MedCPTEmbedder
from src.query.query_analyzer import QueryAnalyzer
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.rrf import RRFManager
from src.storage.parent_store import ParentStore
from src.storage.weaviate_client import AsyncWeaviateChildStore

TEST_PATH = settings.DATA_PATH / "test" / "test_medaesqa.jsonl"
OUTPUT_DIR = settings.TEST_RESULTS_PATH / "medaesqa" / "baseline_vector" / "retrieval"
K_VALUES = settings.K_VALUES


# Per-question metric (P/R/F1 only — reuses retrieval_common primitives)

def compute_prf_metrics(retrieved_pmids: List[str], gold_pmids: Set[str]) -> Dict[str, float]:
    """Compute Precision, Recall, and F1 at each K for a single query."""
    metrics: Dict[str, float] = {}
    for k in K_VALUES:
        p = precision_at_k(retrieved_pmids, gold_pmids, k)
        r = recall_at_k(retrieved_pmids, gold_pmids, k)
        metrics[f"precision_at_{k}"] = round(p, 4)
        metrics[f"recall_at_{k}"] = round(r, 4)
        metrics[f"f1_at_{k}"] = round(f1_score(p, r), 4)
    return metrics


# Main evaluation

async def evaluate(limit: Optional[int] = None) -> None:
    """Run PMID-level retrieval evaluation on the full MedAESQA test set."""
    if not TEST_PATH.exists():
        logger.error(f"Evaluation file not found: {TEST_PATH}")
        sys.exit(1)

    questions: List[Dict[str, Any]] = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    if limit:
        questions = questions[:limit]

    logger.info(
        f"Loaded {len(questions)} questions for MedAESQA retrieval evaluation "
        f"(limit={limit})."
    )
    logger.info(
        f"Runtime settings: "
        f"VECTOR_TOP_K={settings.VECTOR_TOP_K}, "
        f"KEYWORD_TOP_K={settings.KEYWORD_TOP_K}, "
        f"CHILD_FETCH_LIMIT={settings.CHILD_FETCH_LIMIT}, "
        f"K_RRF={settings.K_RRF}, "
        f"TOP_K_RRF={settings.TOP_K_RRF}, "
        f"RERANK_TEXT_TOP_M={settings.RERANK_TEXT_TOP_M}"
    )

    logger.info("Initialising pipeline components...")
    query_analyzer = QueryAnalyzer()
    query_analyzer.temperature = 0.0  # deterministic rewriting
    query_embedder = MedCPTEmbedder(mode="query")
    weaviate_store = AsyncWeaviateChildStore()
    parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)
    rrf_manager = RRFManager()
    cross_encoder = CrossEncoderReranker()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_DIR / "detail.jsonl"
    summary_path = OUTPUT_DIR / "summary.json"

    all_metrics: List[Dict[str, float]] = []
    failed_questions = 0

    try:
        with open(detail_path, "w", encoding="utf-8") as detail_file:
            for i, q in enumerate(questions):
                q_id = q["id"]
                body = q["body"]
                gold_pmids: Set[str] = set(q.get("relevant_pmid", []))

                logger.info(
                    f"[{i + 1}/{len(questions)}] Evaluating Q-{q_id}: {body[:80]}..."
                )

                if not gold_pmids:
                    logger.warning(f"Question {q_id} has no relevant_pmid — skipping.")
                    failed_questions += 1
                    continue

                try:
                    ranked_text, rewritten_query = await run_retrieval_pipeline(
                        query=body,
                        query_analyzer=query_analyzer,
                        query_embedder=query_embedder,
                        weaviate_store=weaviate_store,
                        parent_store=parent_store,
                        rrf_manager=rrf_manager,
                        cross_encoder=cross_encoder,
                        debug_label=q_id,
                        use_vector=True,
                        use_bm25=False,
                    )

                    # Deduplicate PMIDs while preserving rank order
                    seen: Set[str] = set()
                    retrieved_pmids: List[str] = []
                    for item in ranked_text:
                        pmid = item.get("pmid", "")
                        if pmid and pmid not in seen:
                            seen.add(pmid)
                            retrieved_pmids.append(pmid)

                    metrics = compute_prf_metrics(retrieved_pmids, gold_pmids)
                    all_metrics.append(metrics)

                    detail_record = {
                        "question_id": q_id,
                        "body": body,
                        "rewritten_query": rewritten_query,
                        "gold_pmids": sorted(gold_pmids),
                        "num_gold_pmids": len(gold_pmids),
                        "retrieved_items": [
                            {
                                "rank": rank,
                                "pmid": item.get("pmid", ""),
                                "title": item.get("title", ""),
                                "is_relevant": item.get("pmid", "") in gold_pmids,
                                "score": item.get(
                                    "cross_encoder_score", item.get("rrf_score", 0.0)
                                ),
                            }
                            for rank, item in enumerate(ranked_text, 1)
                        ],
                        "metrics": metrics,
                    }
                    detail_file.write(json.dumps(detail_record, ensure_ascii=False) + "\n")
                    detail_file.flush()

                except Exception:
                    logger.exception(f"Failed on question {q_id}:")
                    failed_questions += 1

        summary = _build_summary(all_metrics, len(questions), failed_questions, limit)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        _print_summary(summary)
        logger.info(f"Results saved → {detail_path}  |  {summary_path}")

    finally:
        await query_analyzer.close()
        query_embedder.close()
        if hasattr(cross_encoder, "close"):
            cross_encoder.close()
        weaviate_store.close()
        parent_store.close()


# Summary helpers

def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _build_summary(
    all_metrics: List[Dict[str, float]],
    total_questions: int,
    failed_questions: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    agg: Dict[str, float] = {}
    for k in K_VALUES:
        agg[f"Precision@{k}"] = _mean([m[f"precision_at_{k}"] for m in all_metrics])
        agg[f"Recall@{k}"] = _mean([m[f"recall_at_{k}"] for m in all_metrics])
        agg[f"F1@{k}"] = _mean([m[f"f1_at_{k}"] for m in all_metrics])

    return {
        "config": {
            "dataset": "MedAESQA",
            "split": "test",
            "data_file": str(TEST_PATH),
            "total_questions": total_questions,
            "evaluated_questions": len(all_metrics),
            "failed_questions": failed_questions,
            "limit": limit,
            "vector_top_k": settings.VECTOR_TOP_K,
            "keyword_top_k": settings.KEYWORD_TOP_K,
            "child_fetch_limit": settings.CHILD_FETCH_LIMIT,
            "k_rrf": settings.K_RRF,
            "top_k_rrf": settings.TOP_K_RRF,
            "rerank_text_top_m": settings.RERANK_TEXT_TOP_M,
            "k_values": K_VALUES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "retrieval_metrics": agg,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    cfg = summary["config"]
    m = summary["retrieval_metrics"]

    print("\n" + "=" * 60)
    print("TEXT RETRIEVAL EVALUATION — MedAESQA TEST")
    print("=" * 60)
    print(
        f"  Questions : {cfg['evaluated_questions']}/{cfg['total_questions']}"
        + (f"  (limit={cfg['limit']})" if cfg["limit"] else "")
        + f"  (failed: {cfg['failed_questions']})"
    )
    print(
        f"  Vector_K={cfg['vector_top_k']}  Keyword_K={cfg['keyword_top_k']}  "
        f"RRF_TopK={cfg['top_k_rrf']}  Rerank_TopM={cfg['rerank_text_top_m']}"
    )
    print()
    print(f"  {'Metric':<12} {'@5':>8} {'@10':>8} {'@20':>8}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8}")
    for metric in ["Precision", "Recall", "F1"]:
        vals = [m.get(f"{metric}@{k}", 0.0) for k in K_VALUES]
        print(f"  {metric:<12} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    setup_logging()
    from scripts.evaluation.shared.config_helper import load_and_apply_config
    load_and_apply_config("retrieval")
    
    parser = build_arg_parser(
        "Evaluate text retrieval on the MedAESQA test set "
        "(PMID-level Precision, Recall, F1)."
    )
    args = parser.parse_args()
    import asyncio
    asyncio.run(evaluate(limit=args.limit))
