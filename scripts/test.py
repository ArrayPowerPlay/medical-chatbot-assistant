"""
Text Retrieval Evaluation on BioASQ Validation Data

Evaluates the full text retrieval pipeline against the BioASQ gold standard:
    QueryAnalyzer (temp=0) → Vector + BM25 → RRF → Cross-Encoder

Metrics computed (all at K=5, 10, 20):
    - Precision@K, Recall@K, F1@K  (document-level, PMID matching)
    - MAP@K  (Mean Average Precision truncated at K)
    - MRR    (Mean Reciprocal Rank)
    - Snippet Recall@K, Snippet Precision@K, Snippet F1@K (snippet-level, text containment)

Outputs:
    data/eval_results/bioasq/detail.jsonl   — per-question results
    data/eval_results/bioasq/summary.json   — aggregate metrics

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --limit 10
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple

# Configure project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from config.logging_config import logger
from src.embeddings.medcpt_embedder import MedCPTEmbedder
from src.storage.weaviate_client import WeaviateChildStore
from src.storage.parent_store import ParentStore
from src.retrieval.vector_search import vector_search
from src.retrieval.keyword_search import keyword_search
from src.reranking.rrf import RRFManager
from src.reranking.cross_encoder import CrossEncoderReranker
from src.query.query_analyzer import QueryAnalyzer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_VALUES = [5, 10, 20]
EVAL_RERANK_TOP_M = max(K_VALUES)      # Override production RERANK_TEXT_TOP_M
CHILD_FETCH_LIMIT = 60                 # Adjust after running experiment_child_ratio.py
VAL_PATH = settings.DATA_PATH / "val" / "val_bioasq.jsonl"
OUTPUT_DIR = settings.DATA_PATH / "eval_results" / "bioasq"


# ===========================================================================
# Metric Functions — Document Level (PMID matching)
# ===========================================================================

def precision_at_k(retrieved_pmids: List[str], gold_pmids: Set[str], k: int) -> float:
    """Compute Precision@K.

    Args:
        retrieved_pmids: Ordered list of retrieved PMIDs (rank 1 first).
        gold_pmids: Set of gold-standard relevant PMIDs.
        k: Cutoff rank.

    Returns:
        Precision value in [0, 1].
    """
    top_k = retrieved_pmids[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for pmid in top_k if pmid in gold_pmids)
    return relevant_count / k


def recall_at_k(retrieved_pmids: List[str], gold_pmids: Set[str], k: int) -> float:
    """Compute Recall@K.

    Args:
        retrieved_pmids: Ordered list of retrieved PMIDs (rank 1 first).
        gold_pmids: Set of gold-standard relevant PMIDs.
        k: Cutoff rank.

    Returns:
        Recall value in [0, 1].
    """
    if not gold_pmids:
        return 0.0
    top_k = retrieved_pmids[:k]
    relevant_count = sum(1 for pmid in top_k if pmid in gold_pmids)
    return relevant_count / len(gold_pmids)


def f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall.

    Args:
        precision: Precision value.
        recall: Recall value.

    Returns:
        F1 value in [0, 1]. Returns 0 if both precision and recall are 0.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision_at_k(
    retrieved_pmids: List[str], gold_pmids: Set[str], k: int
) -> float:
    """Compute Average Precision truncated at rank K (AP@K).

    AP@K = (1/min(K, |gold|)) * sum_{i=1}^{K} P(i) * rel(i)

    Args:
        retrieved_pmids: Ordered list of retrieved PMIDs (rank 1 first).
        gold_pmids: Set of gold-standard relevant PMIDs.
        k: Cutoff rank.

    Returns:
        AP@K value in [0, 1].
    """
    if not gold_pmids:
        return 0.0

    top_k = retrieved_pmids[:k]
    relevant_so_far = 0
    precision_sum = 0.0

    for i, pmid in enumerate(top_k, 1):
        if pmid in gold_pmids:
            relevant_so_far += 1
            precision_sum += relevant_so_far / i

    return precision_sum / min(k, len(gold_pmids))


def reciprocal_rank(retrieved_pmids: List[str], gold_pmids: Set[str]) -> float:
    """Compute Reciprocal Rank (RR).

    RR = 1 / rank of the first relevant document. Returns 0 if no relevant
    document is found.

    Args:
        retrieved_pmids: Ordered list of retrieved PMIDs (rank 1 first).
        gold_pmids: Set of gold-standard relevant PMIDs.

    Returns:
        RR value in [0, 1].
    """
    for i, pmid in enumerate(retrieved_pmids, 1):
        if pmid in gold_pmids:
            return 1.0 / i
    return 0.0


def compute_document_metrics(
    retrieved_pmids: List[str], gold_pmids: Set[str]
) -> Dict[str, float]:
    """Compute all document-level metrics for a single query.

    Args:
        retrieved_pmids: Ordered list of retrieved PMIDs (rank 1 first).
        gold_pmids: Set of gold-standard relevant PMIDs.

    Returns:
        Dictionary with metric names as keys and scores as values.
    """
    metrics = {}
    for k in K_VALUES:
        p = precision_at_k(retrieved_pmids, gold_pmids, k)
        r = recall_at_k(retrieved_pmids, gold_pmids, k)
        f1 = f1_score(p, r)
        ap = average_precision_at_k(retrieved_pmids, gold_pmids, k)

        metrics[f"precision_at_{k}"] = round(p, 4)
        metrics[f"recall_at_{k}"] = round(r, 4)
        metrics[f"f1_at_{k}"] = round(f1, 4)
        metrics[f"average_precision_at_{k}"] = round(ap, 4)

    metrics["reciprocal_rank"] = round(reciprocal_rank(retrieved_pmids, gold_pmids), 4)
    return metrics


# ===========================================================================
# Metric Functions — Snippet Level (text containment)
# ===========================================================================

def compute_snippet_metrics(
    retrieved_items: List[Dict],
    gold_snippets: List[Dict],
    gold_pmids: Set[str]
) -> Dict[str, float]:
    """Compute snippet-level recall and precision at each K.

    A gold snippet is considered 'covered' if its text appears as a substring
    within any retrieved parent chunk that shares the same PMID.

    Args:
        retrieved_items: Ordered list of retrieved parent chunks, each with
            'pmid' and 'text' keys.
        gold_snippets: List of gold snippets, each with 'text' and 'pmid' keys.
        gold_pmids: Set of gold-standard relevant PMIDs.

    Returns:
        Dictionary with snippet_recall_at_K, snippet_precision_at_K, and snippet_f1_at_K.
    """
    if not gold_snippets:
        return {f"snippet_recall_at_{k}": 0.0 for k in K_VALUES} | \
               {f"snippet_precision_at_{k}": 0.0 for k in K_VALUES} | \
               {f"snippet_f1_at_{k}": 0.0 for k in K_VALUES}

    total_gold = len(gold_snippets)
    metrics = {}

    for k in K_VALUES:
        top_k_items = retrieved_items[:k]

        # Build a lookup: pmid -> list of retrieved texts
        pmid_to_texts: Dict[str, List[str]] = {}
        for item in top_k_items:
            pmid = item.get("pmid", "")
            text = item.get("text", "")
            if pmid and text and pmid in gold_pmids:
                pmid_to_texts.setdefault(pmid, []).append(text)

        # Count how many gold snippets are covered
        matched = 0
        for snippet in gold_snippets:
            s_pmid = snippet["pmid"]
            s_text = snippet["text"]
            if s_pmid in pmid_to_texts:
                for retrieved_text in pmid_to_texts[s_pmid]:
                    if s_text in retrieved_text:
                        matched += 1
                        break  # Each snippet counted at most once

        snippet_recall = matched / total_gold if total_gold > 0 else 0.0

        # Snippet precision: how many of the retrieved items contain
        # at least one gold snippet
        items_with_snippet = 0
        for item in top_k_items:
            pmid = item.get("pmid", "")
            text = item.get("text", "")
            if pmid in gold_pmids:
                for snippet in gold_snippets:
                    if snippet["pmid"] == pmid and snippet["text"] in text:
                        items_with_snippet += 1
                        break

        snippet_precision = items_with_snippet / k if k > 0 else 0.0
        snippet_f1 = f1_score(snippet_precision, snippet_recall)

        metrics[f"snippet_recall_at_{k}"] = round(snippet_recall, 4)
        metrics[f"snippet_precision_at_{k}"] = round(snippet_precision, 4)
        metrics[f"snippet_f1_at_{k}"] = round(snippet_f1, 4)

    return metrics


# ===========================================================================
# Pipeline Execution
# ===========================================================================

def run_retrieval_pipeline(
    query: str,
    query_analyzer: QueryAnalyzer,
    query_embedder: MedCPTEmbedder,
    weaviate_store: WeaviateChildStore,
    parent_store: ParentStore,
    rrf_manager: RRFManager,
    cross_encoder: CrossEncoderReranker,
) -> Tuple[List[Dict], str]:
    """Run the full text retrieval pipeline for a single query.

    Pipeline: QueryAnalyzer → Vector + BM25 → RRF → Cross-Encoder.

    Args:
        query: Raw user question string.
        query_analyzer: QueryAnalyzer instance for query rewriting.
        query_embedder: MedCPT Query-Encoder for embedding queries.
        weaviate_store: Weaviate client for vector and BM25 search.
        parent_store: SQLite parent store for parent chunk lookup.
        rrf_manager: RRF fusion manager.
        cross_encoder: Cross-Encoder reranker.

    Returns:
        Tuple of (list of parent chunk dicts sorted by rank, rewritten query).
        Each dict has keys: parent_id, pmid, text, title, score/rrf_score,
        cross_encoder_score, source_type.
    """
    # Step 1: Query analysis (temperature=0 for determinism)
    analysis = query_analyzer.analyze(query=query, history=None)
    rewritten_query = analysis.get("rewritten_query", query)

    # Step 2: Encode query
    query_vector = query_embedder.embed_texts(rewritten_query)[0]

    # Step 3: Vector search
    vec_results = vector_search(
        query_vector=query_vector,
        weaviate_store=weaviate_store,
        parent_store=parent_store,
        top_k=EVAL_RERANK_TOP_M,
        child_fetch_limit=CHILD_FETCH_LIMIT,
    )

    # Step 4: BM25 search
    bm25_results = keyword_search(
        query_text=rewritten_query,
        weaviate_store=weaviate_store,
        parent_store=parent_store,
        top_k=EVAL_RERANK_TOP_M,
        child_fetch_limit=CHILD_FETCH_LIMIT,
    )

    # Step 5: RRF fusion
    rrf_results = rrf_manager.rank_fusion(
        vector_results=vec_results,
        bm25_results=bm25_results,
        top_k=50,
    )

    # Step 6: Cross-Encoder reranking (text only, no KG)
    ranked_text, _ = cross_encoder.rerank(
        query=rewritten_query,
        rrf_results=rrf_results,
        kg_results=[],
        top_m=EVAL_RERANK_TOP_M,
        top_n=0,
    )

    return ranked_text, rewritten_query


# ===========================================================================
# Evaluation Loop
# ===========================================================================

def evaluate(limit: int | None = None) -> None:
    """Run the full retrieval evaluation on BioASQ validation data.

    Iterates through each question, runs the pipeline, computes per-question
    metrics, and writes detail + summary output files.

    Args:
        limit: If set, only evaluate the first N questions (for testing).
    """
    # Validate input
    if not VAL_PATH.exists():
        logger.error(f"Validation file not found: {VAL_PATH}")
        sys.exit(1)

    # Load questions
    questions = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    if limit:
        questions = questions[:limit]
    logger.info(f"Loaded {len(questions)} questions for evaluation.")

    # Initialize pipeline components
    logger.info("Initializing pipeline components...")
    query_analyzer = QueryAnalyzer()
    # Override temperature for deterministic evaluation
    query_analyzer.temperature = 0.0
    query_embedder = MedCPTEmbedder(mode="query")
    weaviate_store = WeaviateChildStore()
    parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)
    rrf_manager = RRFManager()
    cross_encoder = CrossEncoderReranker()

    # Prepare output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_DIR / "detail.jsonl"
    summary_path = OUTPUT_DIR / "summary.json"

    # Collect aggregate metrics
    all_doc_metrics: List[Dict[str, float]] = []
    all_snippet_metrics: List[Dict[str, float]] = []
    failed_questions = 0

    try:
        with open(detail_path, "w", encoding="utf-8") as detail_file:
            for i, q in enumerate(questions):
                q_id = q["id"]
                body = q["body"]
                gold_pmids = set(q["relevant_pmid"])
                gold_snippets = q.get("snippets", [])

                logger.info(f"[{i+1}/{len(questions)}] Evaluating: {body[:80]}...")

                try:
                    # Run pipeline
                    ranked_text, rewritten_query = run_retrieval_pipeline(
                        query=body,
                        query_analyzer=query_analyzer,
                        query_embedder=query_embedder,
                        weaviate_store=weaviate_store,
                        parent_store=parent_store,
                        rrf_manager=rrf_manager,
                        cross_encoder=cross_encoder,
                    )

                    # Extract ordered PMIDs (deduplicated, preserving rank order)
                    seen_pmids: Set[str] = set()
                    retrieved_pmids: List[str] = []
                    for item in ranked_text:
                        pmid = item.get("pmid", "")
                        if pmid and pmid not in seen_pmids:
                            seen_pmids.add(pmid)
                            retrieved_pmids.append(pmid)

                    # Compute document-level metrics
                    doc_metrics = compute_document_metrics(retrieved_pmids, gold_pmids)
                    all_doc_metrics.append(doc_metrics)

                    # Compute snippet-level metrics
                    snippet_metrics = compute_snippet_metrics(
                        ranked_text, gold_snippets, gold_pmids
                    )
                    all_snippet_metrics.append(snippet_metrics)

                    # Build per-question detail record
                    retrieved_items_output = []
                    for rank, item in enumerate(ranked_text, 1):
                        retrieved_items_output.append({
                            "rank": rank,
                            "parent_id": item.get("parent_id", ""),
                            "pmid": item.get("pmid", ""),
                            "title": item.get("title", ""),
                            "text": item.get("text", "")[:500],  # Truncate for readability
                            "is_relevant": item.get("pmid", "") in gold_pmids,
                            "score": item.get("cross_encoder_score",
                                              item.get("rrf_score", 0.0)),
                        })

                    detail_record = {
                        "question_id": q_id,
                        "body": body,
                        "rewritten_query": rewritten_query,
                        "gold_pmids": list(gold_pmids),
                        "num_gold_pmids": len(gold_pmids),
                        "retrieved_items": retrieved_items_output,
                        "doc_metrics": doc_metrics,
                        "snippet_metrics": snippet_metrics,
                    }
                    detail_file.write(json.dumps(detail_record, ensure_ascii=False) + "\n")
                    detail_file.flush()

                except Exception as e:
                    logger.error(f"Failed on question {q_id}: {e}")
                    failed_questions += 1
                    continue

        # Aggregate metrics
        summary = _build_summary(
            all_doc_metrics, all_snippet_metrics,
            len(questions), failed_questions
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Print summary table
        _print_summary(summary)
        logger.info(f"Results saved to {detail_path} and {summary_path}")

    finally:
        weaviate_store.close()
        parent_store.close()


def _build_summary(
    all_doc_metrics: List[Dict[str, float]],
    all_snippet_metrics: List[Dict[str, float]],
    total_questions: int,
    failed_questions: int,
) -> Dict[str, Any]:
    """Aggregate per-question metrics into a summary dictionary.

    Args:
        all_doc_metrics: List of document-level metric dicts (one per question).
        all_snippet_metrics: List of snippet-level metric dicts (one per question).
        total_questions: Total number of questions attempted.
        failed_questions: Number of questions that failed.

    Returns:
        Summary dictionary with config and aggregate metrics.
    """
    n = len(all_doc_metrics)

    def mean_metric(metrics_list: List[Dict], key: str) -> float:
        vals = [m[key] for m in metrics_list if key in m]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    # Document-level aggregates
    doc_agg = {}
    for k in K_VALUES:
        doc_agg[f"Precision@{k}"] = mean_metric(all_doc_metrics, f"precision_at_{k}")
        doc_agg[f"Recall@{k}"] = mean_metric(all_doc_metrics, f"recall_at_{k}")
        doc_agg[f"F1@{k}"] = mean_metric(all_doc_metrics, f"f1_at_{k}")
        doc_agg[f"MAP@{k}"] = mean_metric(all_doc_metrics, f"average_precision_at_{k}")
    doc_agg["MRR"] = mean_metric(all_doc_metrics, "reciprocal_rank")

    # Snippet-level aggregates
    snippet_agg = {}
    for k in K_VALUES:
        snippet_agg[f"Snippet_Recall@{k}"] = mean_metric(
            all_snippet_metrics, f"snippet_recall_at_{k}"
        )
        snippet_agg[f"Snippet_Precision@{k}"] = mean_metric(
            all_snippet_metrics, f"snippet_precision_at_{k}"
        )
        snippet_agg[f"Snippet_F1@{k}"] = mean_metric(
            all_snippet_metrics, f"snippet_f1_at_{k}"
        )

    return {
        "config": {
            "val_file": str(VAL_PATH),
            "total_questions": total_questions,
            "evaluated_questions": n,
            "failed_questions": failed_questions,
            "eval_rerank_top_m": EVAL_RERANK_TOP_M,
            "child_fetch_limit": CHILD_FETCH_LIMIT,
            "k_values": K_VALUES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "document_metrics": doc_agg,
        "snippet_metrics": snippet_agg,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted summary table to stdout.

    Args:
        summary: Summary dictionary from _build_summary.
    """
    print("\n" + "=" * 60)
    print("TEXT RETRIEVAL EVALUATION — BioASQ")
    print("=" * 60)

    config = summary["config"]
    print(f"  Questions: {config['evaluated_questions']}/{config['total_questions']}"
          f"  (failed: {config['failed_questions']})")
    print(f"  Rerank Top-M: {config['eval_rerank_top_m']}")
    print(f"  Child Fetch Limit: {config['child_fetch_limit']}")
    print()

    # Document metrics table
    print("DOCUMENT-LEVEL METRICS (PMID matching)")
    print("-" * 60)
    print(f"  {'Metric':<16} {'@5':>8} {'@10':>8} {'@20':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}")

    doc = summary["document_metrics"]
    for metric_name in ["Precision", "Recall", "F1", "MAP"]:
        vals = [doc.get(f"{metric_name}@{k}", 0.0) for k in K_VALUES]
        print(f"  {metric_name:<16} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")

    print(f"  {'MRR':<16} {doc.get('MRR', 0.0):>8.4f}")
    print()

    # Snippet metrics table
    print("SNIPPET-LEVEL METRICS (text containment)")
    print("-" * 60)
    print(f"  {'Metric':<20} {'@5':>8} {'@10':>8} {'@20':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    snip = summary["snippet_metrics"]
    for metric_name in ["Snippet_Recall", "Snippet_Precision", "Snippet_F1"]:
        vals = [snip.get(f"{metric_name}@{k}", 0.0) for k in K_VALUES]
        print(f"  {metric_name:<20} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")

    print("=" * 60)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate text retrieval pipeline on BioASQ validation data."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N questions (for quick testing)."
    )
    args = parser.parse_args()
    evaluate(limit=args.limit)
