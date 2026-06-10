"""
Shared BioASQ generation evaluation utilities for validation/test scripts.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import logger
from config.settings import settings


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
DEFAULT_ROUGE_SKIP_DISTANCE = 4

# Regex to strip common LLM preamble phrases that are absent from BioASQ gold
# references and would reduce ROUGE-SU4 lexical overlap.
_PREAMBLE_RE = re.compile(
    r"^\s*("
    r"based on (the )?(provided |given )?(context|information|documents?|sources?)"
    r"|according to (the )?(provided |given )?(context|documents?|sources?|literature)"
    r"|from (the )?(provided |given )?(context|documents?|information|sources?)"
    r"|the (provided |given )?(context|documents?|information|sources?) (indicate[sd]?|suggest[sd]?|show[sd]?|state[sd]?|mention[sd]?)"
    r"|as (stated|mentioned|described|noted|indicated) in (the )?(provided |given )?(context|documents?|sources?)"
    r")",
    re.IGNORECASE,
)
_PREAMBLE_COMMA_RE = re.compile(r"^[,\.;:]\s*")


def strip_preamble(text: str) -> str:
    """Remove common LLM preamble phrases before computing ROUGE-SU4.

    These phrases (e.g. 'Based on the provided context, ...') do not appear
    in BioASQ gold ideal answers and add noise tokens that hurt recall.
    """
    cleaned = _PREAMBLE_RE.sub("", text.strip())
    # Remove any leading punctuation left after the phrase is stripped.
    cleaned = _PREAMBLE_COMMA_RE.sub("", cleaned)
    return cleaned.strip() or text.strip()  # fall back if everything was stripped


def normalize_text(text: str) -> str:
    """Lightweight normalization for lexical metrics."""
    return " ".join(text.strip().lower().split())


def tokenize(text: str) -> List[str]:
    """Tokenize text for ROUGE-SU4 using a simple alphanumeric regex."""
    return TOKEN_PATTERN.findall(normalize_text(text))


def _build_skip_bigrams(
    tokens: Sequence[str],
    max_skip_distance: int = DEFAULT_ROUGE_SKIP_DISTANCE,
) -> Counter[Tuple[str, ...]]:
    """Build skip-bigram counts with the SU4 max skip constraint."""
    counts: Counter[Tuple[str, ...]] = Counter()
    max_offset = max_skip_distance + 1

    for i, first in enumerate(tokens):
        upper = min(len(tokens), i + max_offset + 1)
        for j in range(i + 1, upper):
            counts[(first, tokens[j])] += 1

    return counts


def _build_su4_units(
    tokens: Sequence[str],
    max_skip_distance: int = DEFAULT_ROUGE_SKIP_DISTANCE,
) -> Counter[Tuple[str, ...]]:
    """Build the combined unigram + skip-bigram multiset for ROUGE-SU4."""
    units: Counter[Tuple[str, ...]] = Counter((token,) for token in tokens)
    units.update(_build_skip_bigrams(tokens, max_skip_distance=max_skip_distance))
    return units


def compute_rouge_su4_single_ref(
    prediction: str,
    reference: str,
    max_skip_distance: int = DEFAULT_ROUGE_SKIP_DISTANCE,
) -> Dict[str, float]:
    """Compute ROUGE-SU4 precision/recall/F1 for one prediction-reference pair."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return {
            "rouge_su4_precision": 0.0,
            "rouge_su4_recall": 0.0,
            "rouge_su4_f1": 0.0,
        }

    pred_units = _build_su4_units(pred_tokens, max_skip_distance=max_skip_distance)
    ref_units = _build_su4_units(ref_tokens, max_skip_distance=max_skip_distance)

    overlap = sum(min(count, ref_units[unit]) for unit, count in pred_units.items())
    pred_total = sum(pred_units.values())
    ref_total = sum(ref_units.values())

    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "rouge_su4_precision": round(precision, 4),
        "rouge_su4_recall": round(recall, 4),
        "rouge_su4_f1": round(f1, 4),
    }


def compute_rouge_su4_multi_ref(
    prediction: str,
    references: Sequence[str],
    max_skip_distance: int = DEFAULT_ROUGE_SKIP_DISTANCE,
) -> Dict[str, float]:
    """Average ROUGE-SU4 across all references for one question."""
    cleaned_refs = [ref.strip() for ref in references if isinstance(ref, str) and ref.strip()]
    if not cleaned_refs:
        return {
            "rouge_su4_precision": 0.0,
            "rouge_su4_recall": 0.0,
            "rouge_su4_f1": 0.0,
        }

    summary: Dict[str, float] = {}
    for key in ["rouge_su4_precision", "rouge_su4_recall", "rouge_su4_f1"]:
        values = [
            compute_rouge_su4_single_ref(
                prediction=prediction,
                reference=reference,
                max_skip_distance=max_skip_distance,
            )[key]
            for reference in cleaned_refs
        ]
        summary[key] = round(sum(values) / len(values), 4)

    return summary


def _summarize_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a JSON-serializable debug view of retrieved sources."""
    summarized: List[Dict[str, Any]] = []

    for rank, item in enumerate(sources, 1):
        summarized.append(
            {
                "rank": rank,
                "source_type": item.get("source_type", "unknown"),
                "pmid": item.get("pmid", ""),
                "title": item.get("title", ""),
                "score": item.get(
                    "cross_encoder_score",
                    item.get("rrf_score", item.get("score", 0.0)),
                ),
                "text": item.get("text", item.get("content", ""))[:700],
                "metadata": item.get("metadata", {}),
            }
        )

    return summarized


def _mean_metric(metrics_list: Iterable[Dict[str, float]], key: str) -> float:
    """Compute a rounded mean for a metric key, filtering out NaN values."""
    import math
    values = [
        metric[key] for metric in metrics_list 
        if key in metric and isinstance(metric[key], (int, float)) and not math.isnan(metric[key])
    ]
    return round(sum(values) / len(values), 4) if values else 0.0


def get_ragas_metric_keys() -> List[str]:
    """Return the fixed set of RAGAS metrics used for generation evaluation."""
    return [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_correctness",
        "answer_relevancy",
    ]


def initialize_ragas_evaluator(enabled: bool) -> Optional[Dict[str, Any]]:
    """Initialize RAGAS runtime components, or log an error and disable it."""
    if not enabled:
        return None

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("RAGAS evaluator requires OPENAI_API_KEY but it is not set.")
        return None

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except Exception as exc:
        logger.error(f"Failed to import RAGAS evaluator dependencies: {exc}")
        return None

    try:
        llm = ChatOpenAI(
            model=settings.RAGAS_EVALUATOR_LLM_MODEL, 
            temperature=settings.GENERATION_TEMPERATURE,
            max_retries=10,
        )
        embeddings = OpenAIEmbeddings(
            model=settings.RAGAS_EVALUATOR_EMBEDDING_MODEL,
            max_retries=10,
        )
    except Exception as exc:
        logger.error(f"Failed to initialize RAGAS evaluator clients: {exc}")
        return None

    return {
        "dataset_cls": Dataset,
        "evaluate_fn": evaluate,
        "llm": llm,
        "embeddings": embeddings,
        "metrics": [
            context_precision,
            context_recall,
            faithfulness,
            answer_correctness,
            answer_relevancy,
        ],
        "metric_keys": get_ragas_metric_keys(),
    }


def evaluate_ragas_question(
    evaluator: Optional[Dict[str, Any]],
    question: str,
    generated_answer: str,
    contexts: List[str],
    references: Sequence[str],
) -> Dict[str, float]:
    """Evaluate one question with RAGAS and average across references."""
    if evaluator is None:
        return {}

    cleaned_refs = [ref.strip() for ref in references if isinstance(ref, str) and ref.strip()]
    if not cleaned_refs:
        return {}

    per_reference_scores: List[Dict[str, float]] = []

    for reference in cleaned_refs:
        dataset = evaluator["dataset_cls"].from_dict(
            {
                "question": [question],
                "answer": [generated_answer],
                "contexts": [contexts],
                "ground_truth": [reference],
            }
        )

        try:
            from ragas.run_config import RunConfig
            run_config = RunConfig(
                max_workers=2,
                max_retries=10,
                max_wait=60,
                timeout=180,
            )
            result = evaluator["evaluate_fn"](
                dataset,
                metrics=evaluator["metrics"],
                llm=evaluator["llm"],
                embeddings=evaluator["embeddings"],
                run_config=run_config,
                raise_exceptions=False,       # If error occurs, continue to run
            )
            row = result.to_pandas().iloc[0].to_dict() # Retrieve the first row and convert to dict
        except Exception as exc:
            logger.error(f"RAGAS evaluation failed for question: {exc}")
            return {}

        scores: Dict[str, float] = {}
        for key in evaluator["metric_keys"]:
            value = row.get(key)
            if isinstance(value, (int, float)):
                scores[key] = round(float(value), 4)
        per_reference_scores.append(scores)

    averaged: Dict[str, float] = {}
    for key in evaluator["metric_keys"]:
        values = [score[key] for score in per_reference_scores if key in score]
        if values:
            averaged[key] = round(sum(values) / len(values), 4)

    return averaged


def build_generation_eval_config(
    split_name: str,
    limit: Optional[int],
    use_ragas: bool,
    kg_top_k: int,
    kg_hop1_m: int,
    kg_hop2_n: int,
    kg_hop2_cap: int,
    rerank_kg_top_n: int,
    generation_temperature: float,
    use_kg_merger: bool,
    use_head_tail_placement: bool,
    generation_max_tokens: int,
) -> Dict[str, Any]:
    """Build the tunable generation-evaluation config dictionary."""
    return {
        "split_name": split_name,
        "limit": limit,
        "requested_ragas": use_ragas,
        "ragas_evaluator_llm_model": settings.RAGAS_EVALUATOR_LLM_MODEL,
        "ragas_evaluator_embedding_model": settings.RAGAS_EVALUATOR_EMBEDDING_MODEL,
        "kg_top_k": kg_top_k,
        "kg_hop1_m": kg_hop1_m,
        "kg_hop2_n": kg_hop2_n,
        "kg_hop2_cap": kg_hop2_cap,
        "rerank_kg_top_n": rerank_kg_top_n,
        "generation_temperature": generation_temperature,
        "use_kg_merger": use_kg_merger,
        "use_head_tail_placement": use_head_tail_placement,
        "generation_max_tokens": generation_max_tokens,
    }


def _print_run_config(config: Dict[str, Any]) -> None:
    """Print the tunable generation-evaluation config to the terminal."""
    print("\n" + "-" * 60)
    print("GENERATION EVAL CONFIG")
    print("-" * 60)
    print(f"  Split: {config['split_name']}")
    print(f"  Limit: {config['limit']}")
    print(f"  Use RAGAS: {config['requested_ragas']}")
    print(f"  RAGAS LLM: {config['ragas_evaluator_llm_model']}")
    print(f"  RAGAS Embeddings: {config['ragas_evaluator_embedding_model']}")
    print(f"  KG top_k: {config['kg_top_k']}")
    print(f"  KG hop1_m: {config['kg_hop1_m']}")
    print(f"  KG hop2_n: {config['kg_hop2_n']}")
    print(f"  KG hop2_cap: {config['kg_hop2_cap']}")
    print(f"  RERANK_KG_TOP_N: {config['rerank_kg_top_n']}")
    print(f"  Temperature: {config['generation_temperature']}")
    print(f"  Use KG merger: {config['use_kg_merger']}")
    print(f"  Use head-tail placement: {config['use_head_tail_placement']}")
    print(f"  Max tokens: {config['generation_max_tokens']}")
    print("-" * 60)


def _close_pipeline(pipeline: Optional[Any]) -> None:
    """Best-effort cleanup for all long-lived clients inside the pipeline."""
    if pipeline is None:
        return

    if hasattr(pipeline, "query_analyzer"):
        pipeline.query_analyzer.close()
    if hasattr(pipeline, "query_embedder"):
        pipeline.query_embedder.close()
    if hasattr(pipeline, "entity_embedder"):
        pipeline.entity_embedder.close()
    if hasattr(pipeline, "cross_encoder_reranker") and hasattr(
        pipeline.cross_encoder_reranker, "close"
    ):
        pipeline.cross_encoder_reranker.close()
    if hasattr(pipeline, "weaviate_store"):
        pipeline.weaviate_store.close()
    if hasattr(pipeline, "parent_store"):
        pipeline.parent_store.close()
    if hasattr(pipeline, "kg_searcher") and hasattr(pipeline.kg_searcher, "cleanup"):
        pipeline.kg_searcher.cleanup()
    if hasattr(pipeline, "llm_generator") and hasattr(pipeline.llm_generator, "client"):
        pipeline.llm_generator.client.close()


def evaluate_split(
    data_path: Path,
    output_dir: Path,
    split_name: str,
    limit: Optional[int] = None,
    use_ragas: bool = True,
    kg_top_k: int = settings.KG_TOP_K,
    kg_hop1_m: int = settings.KG_HOP1_M,
    kg_hop2_n: int = settings.KG_HOP2_N,
    kg_hop2_cap: int = settings.KG_HOP2_CAP,
    rerank_kg_top_n: int = settings.RERANK_KG_TOP_N,
    generation_temperature: float = settings.GENERATION_TEMPERATURE,
    use_kg_merger: bool = settings.USE_KG_MERGER,
    use_head_tail_placement: bool = settings.USE_HEAD_TAIL_PLACEMENT,
    generation_max_tokens: int = settings.GENERATION_MAX_TOKENS,
) -> None:
    """Run generation evaluation for one BioASQ split."""
    if not data_path.exists():
        logger.error(f"Evaluation file not found: {data_path}")
        sys.exit(1)

    questions: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    if limit:
        questions = questions[:limit]

    eval_config = build_generation_eval_config(
        split_name=split_name,
        limit=limit,
        use_ragas=use_ragas,
        kg_top_k=kg_top_k,
        kg_hop1_m=kg_hop1_m,
        kg_hop2_n=kg_hop2_n,
        kg_hop2_cap=kg_hop2_cap,
        rerank_kg_top_n=rerank_kg_top_n,
        generation_temperature=generation_temperature,
        use_kg_merger=use_kg_merger,
        use_head_tail_placement=use_head_tail_placement,
        generation_max_tokens=generation_max_tokens,
    )

    logger.info(f"Loaded {len(questions)} questions for {split_name} generation evaluation.")
    logger.info("Generation eval config: " + json.dumps(eval_config, ensure_ascii=False))
    _print_run_config(eval_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "detail.jsonl"
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.jsonl"

    ragas_evaluator = initialize_ragas_evaluator(enabled=use_ragas)
    if ragas_evaluator is not None:
        logger.info("RAGAS evaluator initialized successfully.")

    pipeline: Optional[Any] = None
    all_generation_metrics: List[Dict[str, float]] = []
    all_ragas_metrics: List[Dict[str, float]] = []
    failed_questions = 0

    try:
        logger.info("Initializing RAG pipeline...")
        from src.pipeline.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.query_analyzer.temperature = 0.0

        with (
            open(detail_path, "w", encoding="utf-8") as detail_file,
            open(predictions_path, "w", encoding="utf-8") as predictions_file,
        ):
            for i, question in enumerate(questions):
                question_id = question["id"]
                body = question["body"]
                gold_pmids = set(question.get("relevant_pmid", []))
                gold_answers = question.get("ideal_answer", [])

                logger.info(
                    f"[{i + 1}/{len(questions)}] Generating answer for question {question_id}"
                )

                try:
                    result = pipeline.run(
                        query=body,
                        history=None,
                        kg_top_k=kg_top_k,
                        kg_hop1_m=kg_hop1_m,
                        kg_hop2_n=kg_hop2_n,
                        kg_hop2_cap=kg_hop2_cap,
                        rerank_kg_top_n=rerank_kg_top_n,
                        generation_temperature=generation_temperature,
                        generation_max_tokens=generation_max_tokens,
                        use_kg_merger=use_kg_merger,
                        use_head_tail_placement=use_head_tail_placement,
                    )
                    generated_answer = result.get("answer", "").strip()
                    rewritten_query = result.get("rewritten_query", body)
                    analysis = result.get("analysis", {})
                    sources = result.get("sources", [])
                    question_type = result.get("question_type", "unknown")

                    # Strip preamble before ROUGE to remove noise tokens absent
                    # from gold references (e.g. 'Based on the provided context...')
                    answer_for_rouge = strip_preamble(generated_answer)
                    generation_metrics = compute_rouge_su4_multi_ref(
                        prediction=answer_for_rouge,
                        references=gold_answers,
                    )
                    all_generation_metrics.append(generation_metrics)

                    ragas_metrics = evaluate_ragas_question(
                        evaluator=ragas_evaluator,
                        question=body,
                        generated_answer=generated_answer,
                        contexts=[
                            text.strip()
                            for item in sources
                            for text in [item.get("text", item.get("content", ""))]
                            if isinstance(text, str) and text.strip()
                        ],
                        references=gold_answers,
                    )
                    if ragas_metrics:
                        all_ragas_metrics.append(ragas_metrics)

                    prediction_record = {
                        "question_id": question_id,
                        "body": body,
                        "rewritten_query": rewritten_query,
                        "generated_answer": generated_answer,
                        "ideal_answer": gold_answers,
                    }
                    predictions_file.write(
                        json.dumps(prediction_record, ensure_ascii=False) + "\n"
                    )
                    predictions_file.flush()

                    detail_record = {
                        "question_id": question_id,
                        "body": body,
                        "question_type": question_type,
                        "rewritten_query": rewritten_query,
                        "analysis": analysis,
                        "generated_answer": generated_answer,
                        "answer_for_rouge": answer_for_rouge,
                        "ideal_answer": gold_answers,
                        "gold_pmids": sorted(gold_pmids),
                        "retrieved_sources": _summarize_sources(sources),
                        "generation_metrics": generation_metrics,
                        "ragas_metrics": ragas_metrics,
                    }
                    detail_file.write(json.dumps(detail_record, ensure_ascii=False) + "\n")
                    detail_file.flush()

                except Exception:
                    logger.exception(f"Failed on question {question_id}:")
                    failed_questions += 1

        summary = _build_summary(
            all_generation_metrics=all_generation_metrics,
            all_ragas_metrics=all_ragas_metrics,
            eval_config=eval_config,
            ragas_enabled=ragas_evaluator is not None,
            ragas_metric_keys=get_ragas_metric_keys(),
            total_questions=len(questions),
            failed_questions=failed_questions,
            data_path=data_path,
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        _print_summary(summary)
        logger.info(
            f"Results saved to {detail_path}, {summary_path}, and {predictions_path}"
        )

    finally:
        _close_pipeline(pipeline)


def _build_summary(
    all_generation_metrics: List[Dict[str, float]],
    all_ragas_metrics: List[Dict[str, float]],
    eval_config: Dict[str, Any],
    ragas_enabled: bool,
    ragas_metric_keys: List[str],
    total_questions: int,
    failed_questions: int,
    data_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Aggregate generation-evaluation outputs into the final summary dict."""
    generation_summary = {
        "ROUGE-SU4-Precision": _mean_metric(all_generation_metrics, "rouge_su4_precision"),
        "ROUGE-SU4-Recall": _mean_metric(all_generation_metrics, "rouge_su4_recall"),
        "ROUGE-SU4-F1": _mean_metric(all_generation_metrics, "rouge_su4_f1"),
    }

    ragas_summary = {
        key: _mean_metric(all_ragas_metrics, key)
        for key in ragas_metric_keys
        if all_ragas_metrics
    }

    return {
        "config": {
            "data_file": str(data_path) if data_path else "",
            "total_questions": total_questions,
            "evaluated_questions": len(all_generation_metrics),
            "failed_questions": failed_questions,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **eval_config,
        },
        "generation_metrics": generation_summary,
        "ragas_metrics": ragas_summary,
        "ragas_enabled": ragas_enabled,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a compact console summary for quick inspection."""
    generation = summary["generation_metrics"]
    ragas_metrics = summary["ragas_metrics"]
    config = summary["config"]

    print("\n" + "=" * 60)
    print(f"GENERATION EVALUATION - BioASQ {config['split_name'].upper()}")
    print("=" * 60)
    print(
        f"  Questions: {config['evaluated_questions']}/{config['total_questions']}"
        f"  (failed: {config['failed_questions']})"
    )
    print(
        f"  KG top_k={config['kg_top_k']}  "
        f"hop1_m={config['kg_hop1_m']}  "
        f"hop2_n={config['kg_hop2_n']}  "
        f"hop2_cap={config['kg_hop2_cap']}"
    )
    print(
        f"  RERANK_KG_TOP_N={config['rerank_kg_top_n']}  "
        f"temperature={config['generation_temperature']}  "
        f"max_tokens={config['generation_max_tokens']}"
    )
    print(
        f"  kg_merger={config['use_kg_merger']}  "
        f"head_tail={config['use_head_tail_placement']}  "
        f"ragas_requested={config['requested_ragas']}  "
        f"ragas_enabled={summary['ragas_enabled']}"
    )
    print(
        f"  ROUGE-SU4: P={generation['ROUGE-SU4-Precision']:.4f}  "
        f"R={generation['ROUGE-SU4-Recall']:.4f}  "
        f"F1={generation['ROUGE-SU4-F1']:.4f}"
    )
    if ragas_metrics:
        print("  RAGAS:")
        for key, value in ragas_metrics.items():
            print(f"    {key}: {value:.4f}")
    print("=" * 60)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build CLI parser for generation evaluation scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (for quick testing).",
    )
    return parser
