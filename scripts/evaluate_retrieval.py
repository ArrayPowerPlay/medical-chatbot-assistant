"""
Text Retrieval Evaluation on BioASQ Validation Data

Evaluates the full text retrieval pipeline against the BioASQ gold standard:
    QueryAnalyzer (temp=0) -> Vector + BM25 -> RRF -> Cross-Encoder

Metrics computed (all at K=5, 10, 20):
    - Precision@K, Recall@K, F1@K  (document-level, PMID matching)
    - MAP@K  (Mean Average Precision truncated at K)
    - MRR    (Mean Reciprocal Rank)
    - Snippet Recall@K, Snippet Precision@K, Snippet F1@K (snippet-level, text containment)

Outputs:
    data/eval_results/bioasq/detail.jsonl   - per-question results
    data/eval_results/bioasq/summary.json   - aggregate metrics
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple, Optional

# Configure project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from config.logging_config import logger
from config.logging_config import setup_logging
from src.embeddings.medcpt_embedder import MedCPTEmbedder
from src.storage.weaviate_client import WeaviateChildStore
from src.storage.parent_store import ParentStore
from src.retrieval.vector_search import vector_search
from src.retrieval.keyword_search import keyword_search
from src.retrieval.parallel_retrieval import ParallelRetriever
from src.reranking.rrf import RRFManager
from src.reranking.cross_encoder import CrossEncoderReranker
from src.query.query_analyzer import QueryAnalyzer


K_VALUES = settings.K_VALUES
CHILD_FETCH_LIMIT = settings.CHILD_FETCH_LIMIT
VAL_PATH = settings.DATA_PATH / "val" / "val_bioasq.jsonl"
OUTPUT_DIR = settings.DATA_PATH / "eval_results" / "bioasq"


def precision_at_k(retrieved_pmids: List[str], gold_pmids: Set[str], k: int) -> float:
    """Compute Precision@K."""
    top_k = retrieved_pmids[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for pmid in top_k if pmid in gold_pmids)
    return relevant_count / k


def recall_at_k(retrieved_pmids: List[str], gold_pmids: Set[str], k: int) -> float:
    """Compute Recall@K."""
    if not gold_pmids:
        return 0.0
    top_k = retrieved_pmids[:k]
    relevant_count = sum(1 for pmid in top_k if pmid in gold_pmids)
    return relevant_count / len(gold_pmids)


def f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision_at_k(
    retrieved_pmids: List[str], gold_pmids: Set[str], k: int
) -> float:
    """Compute Average Precision truncated at rank K (AP@K)."""
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
    """Compute Reciprocal Rank (RR)."""
    for i, pmid in enumerate(retrieved_pmids, 1):
        if pmid in gold_pmids:
            return 1.0 / i
    return 0.0


def compute_document_metrics(
    retrieved_pmids: List[str], gold_pmids: Set[str]
) -> Dict[str, float]:
    """Compute all document-level metrics for a single query."""
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


def compute_snippet_metrics(
    retrieved_items: List[Dict],
    gold_snippets: List[Dict],
    gold_pmids: Set[str]
) -> Dict[str, float]:
    """Compute snippet-level recall and precision at each K."""
    if not gold_snippets:
        return {f"snippet_recall_at_{k}": 0.0 for k in K_VALUES} | {
            f"snippet_precision_at_{k}": 0.0 for k in K_VALUES
        } | {f"snippet_f1_at_{k}": 0.0 for k in K_VALUES}

    total_gold = len(gold_snippets)
    metrics = {}

    for k in K_VALUES:
        top_k_items = retrieved_items[:k]

        pmid_to_texts: Dict[str, List[str]] = {}
        for item in top_k_items:
            pmid = item.get("pmid", "")
            text = item.get("text", "")
            if pmid and text and pmid in gold_pmids:
                pmid_to_texts.setdefault(pmid, []).append(text)

        matched = 0
        for snippet in gold_snippets:
            s_pmid = snippet["pmid"]
            s_text = snippet["text"]
            if s_pmid in pmid_to_texts:
                for retrieved_text in pmid_to_texts[s_pmid]:
                    if s_text in retrieved_text:
                        matched += 1
                        break

        snippet_recall = matched / total_gold if total_gold > 0 else 0.0

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


def run_retrieval_pipeline(
    query: str,
    query_analyzer: QueryAnalyzer,
    query_embedder: MedCPTEmbedder,
    weaviate_store: WeaviateChildStore,
    parent_store: ParentStore,
    rrf_manager: RRFManager,
    cross_encoder: CrossEncoderReranker,
    debug_label: Optional[str] = None,
) -> Tuple[List[Dict], str]:
    """Run the full text retrieval pipeline for a single query."""
    label = f"[{debug_label}] " if debug_label else ""

    logger.info(f"{label}QueryAnalyzer starting")
    analysis = query_analyzer.analyze(query=query, history=None)
    rewritten_query = analysis.get("rewritten_query", query)
    logger.info(f"{label}QueryAnalyzer done. rewritten_query={rewritten_query!r}")

    logger.info(f"{label}Embedding query")
    query_vector = query_embedder.embed_texts(rewritten_query)[0]

    logger.info(
        f"{label}Vector search starting with top_k={settings.VECTOR_TOP_K}, "
        f"child_fetch_limit={CHILD_FETCH_LIMIT}"
    )
    vec_results = vector_search(
        query_vector=query_vector,
        weaviate_store=weaviate_store,
        parent_store=parent_store,
        top_k=settings.VECTOR_TOP_K,
        child_fetch_limit=CHILD_FETCH_LIMIT,
    )
    logger.info(f"{label}Vector search returned {len(vec_results)} parent results")

    logger.info(
        f"{label}BM25 search starting with top_k={settings.KEYWORD_TOP_K}, "
        f"child_fetch_limit={CHILD_FETCH_LIMIT}"
    )
    bm25_results = keyword_search(
        query_text=rewritten_query,
        weaviate_store=weaviate_store,
        parent_store=parent_store,
        top_k=settings.KEYWORD_TOP_K,
        child_fetch_limit=CHILD_FETCH_LIMIT,
    )
    logger.info(f"{label}BM25 search returned {len(bm25_results)} parent results")

    logger.info(f"{label}RRF fusion starting with k={rrf_manager.k}")
    rrf_results = rrf_manager.rank_fusion(
        vector_results=vec_results,
        bm25_results=bm25_results,
        top_k=settings.TOP_K_RRF,
    )
    logger.info(f"{label}RRF fusion returned {len(rrf_results)} fused results")

    logger.info(f"{label}Cross-encoder rerank starting")
    ranked_text, _ = cross_encoder.rerank(
        query=rewritten_query,
        rrf_results=rrf_results,
        kg_results=[],
        top_m=settings.RERANK_TEXT_TOP_M,
        top_n=0,
    )
    logger.info(f"{label}Cross-encoder rerank returned {len(ranked_text)} text results")

    return ranked_text, rewritten_query


def evaluate(limit: int | None = None, question_id: Optional[str] = None) -> None:
    """Run the full retrieval evaluation on BioASQ validation data."""
    if not VAL_PATH.exists():
        logger.error(f"Validation file not found: {VAL_PATH}")
        sys.exit(1)

    questions = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    if question_id:
        questions = [q for q in questions if q["id"] == question_id]
        if not questions:
            logger.error(f"Question ID not found in validation file: {question_id}")
            sys.exit(1)

    if limit:
        questions = questions[:limit]
    logger.info(f"Loaded {len(questions)} questions for evaluation.")
    logger.info(
        "Runtime settings: "
        f"K_RRF={settings.K_RRF}, "
        f"TOP_K_RRF={settings.TOP_K_RRF}, "
        f"VECTOR_TOP_K={settings.KEYWORD_TOP_K}, "
        f"KEYWORD_TOP_K={settings.KEYWORD_TOP_K}, "
        f"CHILD_FETCH_LIMIT={settings.CHILD_FETCH_LIMIT}, "
        f"RERANK_TEXT_TOP_M={settings.RERANK_TEXT_TOP_M}, "
        f"K_VALUES={settings.K_VALUES}"
    )

    logger.info("Initializing pipeline components...")
    query_analyzer = QueryAnalyzer()
    query_analyzer.temperature = 0.0
    query_embedder = MedCPTEmbedder(mode="query")
    weaviate_store = WeaviateChildStore()
    parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)
    rrf_manager = RRFManager()
    cross_encoder = CrossEncoderReranker()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_DIR / "detail.jsonl"
    summary_path = OUTPUT_DIR / "summary.json"

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
                    ranked_text, rewritten_query = run_retrieval_pipeline(
                        query=body,
                        query_analyzer=query_analyzer,
                        query_embedder=query_embedder,
                        weaviate_store=weaviate_store,
                        parent_store=parent_store,
                        rrf_manager=rrf_manager,
                        cross_encoder=cross_encoder,
                        debug_label=q_id,
                    )

                    seen_pmids: Set[str] = set()
                    retrieved_pmids: List[str] = []
                    for item in ranked_text:
                        pmid = item.get("pmid", "")
                        if pmid and pmid not in seen_pmids:
                            seen_pmids.add(pmid)
                            retrieved_pmids.append(pmid)

                    doc_metrics = compute_document_metrics(retrieved_pmids, gold_pmids)
                    all_doc_metrics.append(doc_metrics)

                    snippet_metrics = compute_snippet_metrics(
                        ranked_text, gold_snippets, gold_pmids
                    )
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

                except Exception:
                    logger.exception(f"Failed on question {q_id}:")
                    failed_questions += 1
                    continue

        summary = _build_summary(
            all_doc_metrics, all_snippet_metrics, len(questions), failed_questions
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        _print_summary(summary)
        logger.info(f"Results saved to {detail_path} and {summary_path}")

    finally:
        query_analyzer.close()
        query_embedder.close()
        if hasattr(cross_encoder, "close"):
            cross_encoder.close()
        weaviate_store.close()
        parent_store.close()


def _build_summary(
    all_doc_metrics: List[Dict[str, float]],
    all_snippet_metrics: List[Dict[str, float]],
    total_questions: int,
    failed_questions: int,
) -> Dict[str, Any]:
    """Aggregate per-question metrics into a summary dictionary."""
    n = len(all_doc_metrics)

    def mean_metric(metrics_list: List[Dict], key: str) -> float:
        vals = [m[key] for m in metrics_list if key in m]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    doc_agg = {}
    for k in K_VALUES:
        doc_agg[f"Precision@{k}"] = mean_metric(all_doc_metrics, f"precision_at_{k}")
        doc_agg[f"Recall@{k}"] = mean_metric(all_doc_metrics, f"recall_at_{k}")
        doc_agg[f"F1@{k}"] = mean_metric(all_doc_metrics, f"f1_at_{k}")
        doc_agg[f"MAP@{k}"] = mean_metric(
            all_doc_metrics, f"average_precision_at_{k}"
        )
    doc_agg["MRR"] = mean_metric(all_doc_metrics, "reciprocal_rank")

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
            "k_rrf": settings.K_RRF,
            "top_k_rrf": settings.TOP_K_RRF,
            "vector_top_k": settings.VECTOR_TOP_K,
            "keyword_top_k": settings.KEYWORD_TOP_K,
            "rerank_text_top_m": settings.RERANK_TEXT_TOP_M,
            "child_fetch_limit": CHILD_FETCH_LIMIT,
            "k_values": K_VALUES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "document_metrics": doc_agg,
        "snippet_metrics": snippet_agg,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 60)
    print("TEXT RETRIEVAL EVALUATION - BioASQ")
    print("=" * 60)

    config = summary["config"]
    print(
        f"  Questions: {config['evaluated_questions']}/{config['total_questions']}"
        f"  (failed: {config['failed_questions']})"
    )
    print(f"  Rerank Top-M: {config['rerank_text_top_m']}")
    print(f"  Child Fetch Limit: {config['child_fetch_limit']}")
    print()

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

    print("SNIPPET-LEVEL METRICS (text containment)")
    print("-" * 60)
    print(f"  {'Metric':<20} {'@5':>8} {'@10':>8} {'@20':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    snip = summary["snippet_metrics"]
    for metric_name in ["Snippet_Recall", "Snippet_Precision", "Snippet_F1"]:
        vals = [snip.get(f"{metric_name}@{k}", 0.0) for k in K_VALUES]
        print(f"  {metric_name:<20} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")

    print("=" * 60)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Evaluate text retrieval pipeline on BioASQ validation data."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N questions (for quick testing)."
    )
    parser.add_argument(
        "--question-id", type=str, default=None,
        help="Evaluate only one specific BioASQ question ID."
    )
    args = parser.parse_args()
    evaluate(limit=args.limit, question_id=args.question_id)
