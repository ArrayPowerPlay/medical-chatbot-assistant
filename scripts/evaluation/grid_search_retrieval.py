"""
Grid search for text retrieval hyperparameters on a 20-question BioASQ subset.

This script evaluates a fixed grid of retrieval configurations and saves
full metrics for each run into a single JSON file.

Default behavior:
    - Evaluate exactly 20 validation questions
    - Tune VECTOR_TOP_K and KEYWORD_TOP_K independently
    - Keep other strong baseline settings fixed unless overridden

Output:
    results/eval_results/bioasq/retrieval/grid_search_20q_<timestamp>.json
"""

import sys
import asyncio
import json
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from config.logging_config import logger, setup_logging
from src.embeddings.medcpt_embedder import MedCPTEmbedder
from src.storage.weaviate_client import AsyncWeaviateChildStore
from src.storage.parent_store import ParentStore
from src.reranking.rrf import RRFManager
from src.reranking.cross_encoder import CrossEncoderReranker
from src.query.query_analyzer import QueryAnalyzer
from scripts.evaluation.shared import retrieval_common as er


VAL_PATH = settings.DATA_PATH / "val" / "val_bioasq.jsonl"
OUTPUT_DIR = settings.EVAL_RESULTS_PATH / "bioasq" / "retrieval"
DEFAULT_LIMIT = 20


DEFAULT_GRID: List[Dict[str, int]] = [
    {"vector_top_k": 40, "keyword_top_k": 20},
    {"vector_top_k": 80, "keyword_top_k": 20},
    {"vector_top_k": 120, "keyword_top_k": 20},
    {"vector_top_k": 40, "keyword_top_k": 40},
    {"vector_top_k": 80, "keyword_top_k": 40},
    {"vector_top_k": 120, "keyword_top_k": 40},
    {"vector_top_k": 40, "keyword_top_k": 80},
    {"vector_top_k": 80, "keyword_top_k": 80},
    {"vector_top_k": 120, "keyword_top_k": 80},
    {"vector_top_k": 20, "keyword_top_k": 80},
    {"vector_top_k": 20, "keyword_top_k": 120},
    {"vector_top_k": 40, "keyword_top_k": 120},
    {"vector_top_k": 80, "keyword_top_k": 120},
]


def load_questions(limit: int) -> List[Dict[str, Any]]:
    """Load the first N validation questions."""
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation file not found: {VAL_PATH}")

    questions: List[Dict[str, Any]] = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    return questions[:limit]


async def evaluate_questions(
    questions: List[Dict[str, Any]],
    query_analyzer: QueryAnalyzer,
    query_embedder: MedCPTEmbedder,
    weaviate_store: AsyncWeaviateChildStore,
    parent_store: ParentStore,
    rrf_manager: RRFManager,
    cross_encoder: CrossEncoderReranker,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate questions and return both aggregate summary and per-question details."""
    all_doc_metrics: List[Dict[str, float]] = []
    all_snippet_metrics: List[Dict[str, float]] = []
    detail_records: List[Dict[str, Any]] = []
    failed_questions = 0

    for i, q in enumerate(questions):
        q_id = q["id"]
        body = q["body"]
        gold_pmids = set(q["relevant_pmid"])
        gold_snippets = q.get("snippets", [])

        logger.info(
            f"[Grid Search][{i+1}/{len(questions)}][{q_id}] "
            f"V={settings.VECTOR_TOP_K} K={settings.KEYWORD_TOP_K}"
        )

        try:
            ranked_text, rewritten_query = await er.run_retrieval_pipeline(
                query=body,
                query_analyzer=query_analyzer,
                query_embedder=query_embedder,
                weaviate_store=weaviate_store,
                parent_store=parent_store,
                rrf_manager=rrf_manager,
                cross_encoder=cross_encoder,
                debug_label=q_id,
            )

            seen_pmids = set()
            retrieved_pmids: List[str] = []
            for item in ranked_text:
                pmid = item.get("pmid", "")
                if pmid and pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    retrieved_pmids.append(pmid)

            doc_metrics = er.compute_document_metrics(retrieved_pmids, gold_pmids)
            snippet_metrics = er.compute_snippet_metrics(
                ranked_text, gold_snippets, gold_pmids
            )

            all_doc_metrics.append(doc_metrics)
            all_snippet_metrics.append(snippet_metrics)

            retrieved_items_output = []
            for rank, item in enumerate(ranked_text, 1):
                retrieved_items_output.append({
                    "rank": rank,
                    "parent_id": item.get("parent_id", ""),
                    "pmid": item.get("pmid", ""),
                    "title": item.get("title", ""),
                    "text": item.get("text", "")[:500],
                    "is_relevant": item.get("pmid", "") in gold_pmids,
                    "score": item.get(
                        "cross_encoder_score",
                        item.get("rrf_score", 0.0),
                    ),
                })

            detail_records.append({
                "question_id": q_id,
                "body": body,
                "rewritten_query": rewritten_query,
                "gold_pmids": list(gold_pmids),
                "num_gold_pmids": len(gold_pmids),
                "retrieved_items": retrieved_items_output,
                "doc_metrics": doc_metrics,
                "snippet_metrics": snippet_metrics,
            })

        except Exception:
            logger.exception(f"[Grid Search] Failed on question {q_id}:")
            failed_questions += 1

    summary = er._build_summary(
        all_doc_metrics=all_doc_metrics,
        all_snippet_metrics=all_snippet_metrics,
        total_questions=len(questions),
        failed_questions=failed_questions,
    )
    return summary, detail_records


def score_run(summary: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Sort key for comparing runs."""
    doc = summary["document_metrics"]
    return (
        doc.get("MAP@10", 0.0),
        doc.get("GMAP@10", 0.0),
        doc.get("MRR", 0.0),
        doc.get("Precision@5", 0.0),
    )


def aggregate_metric_columns(run: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Flatten aggregate summary metrics into ordered columns for wide terminal output."""
    doc = run["summary"]["document_metrics"]
    snip = run["summary"]["snippet_metrics"]
    ordered_keys = [
        "Precision@5", "Recall@5", "F1@5", "MAP@5", "GMAP@5",
        "Precision@10", "Recall@10", "F1@10", "MAP@10", "GMAP@10",
        "Precision@20", "Recall@20", "F1@20", "MAP@20", "GMAP@20",
        "MRR",
        "Snippet_Recall@5", "Snippet_Precision@5", "Snippet_F1@5",
        "Snippet_Recall@10", "Snippet_Precision@10", "Snippet_F1@10",
        "Snippet_Recall@20", "Snippet_Precision@20", "Snippet_F1@20",
    ]
    values: List[Tuple[str, float]] = []
    for key in ordered_keys:
        if key in doc:
            values.append((key, doc[key]))
        elif key in snip:
            values.append((key, snip[key]))
    return values


def apply_run_settings(
    vector_top_k: int,
    keyword_top_k: int,
    child_fetch_limit: int,
    top_k_rrf: int,
    k_rrf: int,
    rerank_text_top_m: int,
) -> None:
    """Mutate runtime settings for one grid-search run."""
    settings.VECTOR_TOP_K = vector_top_k
    settings.KEYWORD_TOP_K = keyword_top_k
    settings.CHILD_FETCH_LIMIT = child_fetch_limit
    settings.TOP_K_RRF = top_k_rrf
    settings.K_RRF = k_rrf
    settings.RERANK_TEXT_TOP_M = rerank_text_top_m


async def run_grid_search(
    child_fetch_limit: int,
    top_k_rrf: int,
    k_rrf: int,
    rerank_text_top_m: int,
) -> Path:
    """Run the default grid and save all config/metric outputs to one JSON file."""
    limit = DEFAULT_LIMIT
    questions = load_questions(limit=limit)
    logger.info(f"Loaded {len(questions)} validation questions for grid search.")

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
    weaviate_store = AsyncWeaviateChildStore()
    parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)
    cross_encoder = CrossEncoderReranker()

    runs: List[Dict[str, Any]] = []

    try:
        for idx, cfg in enumerate(DEFAULT_GRID, 1):
            apply_run_settings(
                vector_top_k=cfg["vector_top_k"],
                keyword_top_k=cfg["keyword_top_k"],
                child_fetch_limit=child_fetch_limit,
                top_k_rrf=top_k_rrf,
                k_rrf=k_rrf,
                rerank_text_top_m=rerank_text_top_m,
            )
            rrf_manager = RRFManager(k=settings.K_RRF)

            logger.info(
                f"[Grid Search] Run {idx}/{len(DEFAULT_GRID)}: "
                f"VECTOR_TOP_K={settings.VECTOR_TOP_K}, "
                f"KEYWORD_TOP_K={settings.KEYWORD_TOP_K}, "
                f"CHILD_FETCH_LIMIT={settings.CHILD_FETCH_LIMIT}, "
                f"TOP_K_RRF={settings.TOP_K_RRF}, "
                f"K_RRF={settings.K_RRF}, "
                f"RERANK_TEXT_TOP_M={settings.RERANK_TEXT_TOP_M}"
            )

            summary, detail_records = await evaluate_questions(
                questions=questions,
                query_analyzer=query_analyzer,
                query_embedder=query_embedder,
                weaviate_store=weaviate_store,
                parent_store=parent_store,
                rrf_manager=rrf_manager,
                cross_encoder=cross_encoder,
            )

            runs.append(
                {
                    "run_index": idx,
                    "config": summary["config"],
                    "summary": summary,
                    "detail_records": detail_records,
                }
            )

        runs.sort(
            key=lambda run: score_run(
                {
                    "document_metrics": run["summary"]["document_metrics"],
                    "snippet_metrics": run["summary"]["snippet_metrics"],
                }
            ),
            reverse=True,
        )

        result = {
            "metadata": {
                "limit": limit,
                "grid_size": len(DEFAULT_GRID),
                "ranking_key": ["MAP@10", "GMAP@10", "MRR", "Precision@5"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "notes": (
                    "Each run reports full summary metrics and per-question detail "
                    "records from scripts/evaluation/bioasq/val_retrieval.py "
                    "on the same 20-question validation subset."
                ),
            },
            "runs": runs,
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"grid_search_20q_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 72)
        print("GRID SEARCH COMPLETE")
        print("=" * 72)
        print(f"Questions: {limit}")
        print(f"Runs: {len(runs)}")
        print(f"Saved: {output_path}")
        print()
        header = ["Rank", "V_TOP_K", "K_TOP_K"]
        metric_names = [name for name, _ in aggregate_metric_columns(runs[0])]
        header.extend(metric_names)
        print(" | ".join(header))
        print("-" * max(72, len(" | ".join(header))))
        for rank, run in enumerate(runs, 1):
            config = run["config"]
            row = [
                str(rank),
                str(config["vector_top_k"]),
                str(config["keyword_top_k"]),
            ]
            row.extend(f"{value:.4f}" for _, value in aggregate_metric_columns(run))
            print(" | ".join(row))

        return output_path

    finally:
        apply_run_settings(
            vector_top_k=original_settings["vector_top_k"],
            keyword_top_k=original_settings["keyword_top_k"],
            child_fetch_limit=original_settings["child_fetch_limit"],
            top_k_rrf=original_settings["top_k_rrf"],
            k_rrf=original_settings["k_rrf"],
            rerank_text_top_m=original_settings["rerank_text_top_m"],
        )
        await query_analyzer.close()
        query_embedder.close()
        if hasattr(cross_encoder, "close"):
            cross_encoder.close()
        weaviate_store.close()
        parent_store.close()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for grid search."""
    parser = argparse.ArgumentParser(
        description="Run a 20-question retrieval grid search and save config + metrics."
    )
    parser.add_argument(
        "--child-fetch-limit",
        type=int,
        default=settings.CHILD_FETCH_LIMIT,
        help="Child fetch limit shared by all runs (default: current settings value).",
    )
    parser.add_argument(
        "--top-k-rrf",
        type=int,
        default=settings.TOP_K_RRF,
        help="RRF output depth shared by all runs (default: current settings value).",
    )
    parser.add_argument(
        "--k-rrf",
        type=int,
        default=settings.K_RRF,
        help="RRF damping constant shared by all runs (default: current settings value).",
    )
    parser.add_argument(
        "--rerank-text-top-m",
        type=int,
        default=settings.RERANK_TEXT_TOP_M,
        help="Cross-encoder top-M shared by all runs (default: current settings value).",
    )
    return parser


if __name__ == "__main__":
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    asyncio.run(run_grid_search(
        child_fetch_limit=args.child_fetch_limit,
        top_k_rrf=args.top_k_rrf,
        k_rrf=args.k_rrf,
        rerank_text_top_m=args.rerank_text_top_m,
    ))
