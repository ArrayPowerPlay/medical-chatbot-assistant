"""
Shared MedAESQA generation evaluation utilities for validation/test scripts.

MedAESQA is used as a secondary citation benchmark. The metric set is:
  - ROUGE-SU4-F1  (cross-comparison with BioASQ)
  - Citation Precision / Recall / F1  (primary MedAESQA signal)
RAGAS is intentionally omitted: it is expensive and redundant given that
BioASQ is the primary evaluation dataset and MedAESQA's unique contribution
is citation attribution quality.

ROUGE note: MedAESQA gold answers embed inline PMID citations such as
[28646811, 33659106] directly within the reference text. These brackets are
stripped from gold references before ROUGE-SU4 computation so that lexical
overlap reflects content quality, not citation-format differences.
The raw (unstripped) gold answer is preserved in the detail output.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import logger
from config.settings import settings
from scripts.evaluation.shared import generation_bioasq_common as bioasq_common

PMID_GROUP_PATTERN = re.compile(r"\[(?:PMID:\s*)?(\d+(?:,\s*\d+)*)\]", re.IGNORECASE)


def extract_pmids_from_text(text: str) -> Set[str]:
    """Extract PMID groups formatted like [12345, 67890] from text."""
    pmids: Set[str] = set()
    for group in PMID_GROUP_PATTERN.findall(text or ""):
        for pmid in group.split(","):
            cleaned = pmid.strip()
            if cleaned:
                pmids.add(cleaned)
    return pmids


def strip_pmid_brackets(text: str) -> str:
    """Remove inline PMID citation brackets from text before ROUGE computation.

    MedAESQA gold answers embed citations like [28646811, 33659106] directly
    in the reference text. Stripping them ensures ROUGE-SU4 measures content
    overlap rather than citation-format differences.

    Example:
        Input:  "Weight loss reduces apnea [33659106, 12693795]."
        Output: "Weight loss reduces apnea."
    """
    cleaned = PMID_GROUP_PATTERN.sub("", text or "")
    # Collapse any double spaces left after removal
    return re.sub(r" {2,}", " ", cleaned).strip()


def compute_citation_metrics(
    prediction: str,
    gold_pmids: Set[str],
) -> Dict[str, float]:
    """Compute PMID citation precision, recall, and F1."""
    cited_pmids = extract_pmids_from_text(prediction)

    if cited_pmids:
        precision = len(cited_pmids & gold_pmids) / len(cited_pmids)
    else:
        precision = 0.0

    if gold_pmids:
        recall = len(cited_pmids & gold_pmids) / len(gold_pmids)
    else:
        recall = 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(f1, 4),
    }


async def evaluate_split(
    data_path: Path,
    output_dir: Path,
    split_name: str,
    limit: Optional[int] = None,
    kg_top_k: int = settings.KG_TOP_K,
    kg_hop1_m: int = settings.KG_HOP1_M,
    kg_hop2_n: int = settings.KG_HOP2_N,
    kg_hop2_cap: int = settings.KG_HOP2_CAP,
    rerank_kg_top_n: int = settings.RERANK_KG_TOP_N,
    generation_temperature: float = settings.GENERATION_TEMPERATURE,
    use_kg_merger: bool = settings.USE_KG_MERGER,
    use_head_tail_placement: bool = settings.USE_HEAD_TAIL_PLACEMENT,
    generation_max_tokens: int = settings.GENERATION_MAX_TOKENS,
    use_citations: bool = settings.USE_CITATIONS,
) -> None:
    """Run generation evaluation for one MedAESQA split."""
    if not data_path.exists():
        logger.error(f"Evaluation file not found: {data_path}")
        sys.exit(1)

    questions: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    if limit:
        questions = questions[:limit]

    eval_config = bioasq_common.build_generation_eval_config(
        split_name=split_name,
        limit=limit,
        use_ragas=False,  # RAGAS not used for MedAESQA (secondary dataset)
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
    eval_config["use_citations"] = use_citations

    logger.info(f"Loaded {len(questions)} questions for MedAESQA {split_name} generation evaluation.")
    logger.info("Generation eval config: " + json.dumps(eval_config, ensure_ascii=False))
    bioasq_common._print_run_config(eval_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "detail.jsonl"
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.jsonl"

    pipeline: Optional[Any] = None
    all_generation_metrics: List[Dict[str, float]] = []
    all_citation_metrics: List[Dict[str, float]] = []
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
                    result = await pipeline.run(
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
                        use_citations=use_citations,
                    )
                    generated_answer = result.get("answer", "").strip()
                    rewritten_query = result.get("rewritten_query", body)
                    analysis = result.get("analysis", {})
                    sources = result.get("sources", [])
                    question_type = result.get("question_type", "unknown")

                    # Strip preamble then PMID brackets from model answer before ROUGE.
                    # ROUGE measures content quality, not citation format:
                    #   - strip_preamble: removes LLM openers ("Based on the context...")
                    #   - strip_pmid_brackets: removes [PMID: 12345] tokens that add
                    #     noise to lexical overlap without reflecting answer quality.
                    # The raw generated_answer (with citations) is preserved for
                    # citation_* metrics, which measure attribution behaviour separately.
                    answer_for_rouge = strip_pmid_brackets(
                        bioasq_common.strip_preamble(generated_answer)
                    )

                    # Strip inline PMID brackets from gold references before ROUGE.
                    # MedAESQA gold answers embed [28646811, 33659106] in the text;
                    # removing them gives a fair content-only comparison on both sides.
                    gold_answers_for_rouge = [strip_pmid_brackets(ref) for ref in gold_answers]

                    generation_metrics = bioasq_common.compute_rouge_su4_multi_ref(
                        prediction=answer_for_rouge,
                        references=gold_answers_for_rouge,
                    )
                    all_generation_metrics.append(generation_metrics)

                    citation_metrics = compute_citation_metrics(
                        prediction=generated_answer,
                        gold_pmids=gold_pmids,
                    )
                    all_citation_metrics.append(citation_metrics)

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
                        "ideal_answer_for_rouge": gold_answers_for_rouge,
                        "gold_pmids": sorted(gold_pmids),
                        "retrieved_sources": bioasq_common._summarize_sources(sources),
                        "generation_metrics": generation_metrics,
                        "citation_metrics": citation_metrics,
                    }
                    detail_file.write(json.dumps(detail_record, ensure_ascii=False) + "\n")
                    detail_file.flush()

                except Exception:
                    logger.exception(f"Failed on question {question_id}:")
                    failed_questions += 1

        summary = _build_summary(
            all_generation_metrics=all_generation_metrics,
            all_citation_metrics=all_citation_metrics,
            eval_config=eval_config,
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
        await bioasq_common._close_pipeline(pipeline)


def _build_summary(
    all_generation_metrics: List[Dict[str, float]],
    all_citation_metrics: List[Dict[str, float]],
    eval_config: Dict[str, Any],
    total_questions: int,
    failed_questions: int,
    data_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Aggregate MedAESQA generation-evaluation outputs into the final summary dict."""
    generation_summary = {
        "ROUGE-SU4-Precision": bioasq_common._mean_metric(all_generation_metrics, "rouge_su4_precision"),
        "ROUGE-SU4-Recall": bioasq_common._mean_metric(all_generation_metrics, "rouge_su4_recall"),
        "ROUGE-SU4-F1": bioasq_common._mean_metric(all_generation_metrics, "rouge_su4_f1"),
    }

    citation_summary = {
        "Citation-Precision": bioasq_common._mean_metric(all_citation_metrics, "citation_precision"),
        "Citation-Recall": bioasq_common._mean_metric(all_citation_metrics, "citation_recall"),
        "Citation-F1": bioasq_common._mean_metric(all_citation_metrics, "citation_f1"),
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
        "citation_metrics": citation_summary,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a compact console summary for MedAESQA evaluation."""
    generation = summary["generation_metrics"]
    citation = summary["citation_metrics"]
    config = summary["config"]

    print("\n" + "=" * 60)
    print(f"GENERATION EVALUATION - MedAESQA {config['split_name'].upper()}")
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
        f"use_citations={config.get('use_citations', True)}"
    )
    print(
        f"  ROUGE-SU4: P={generation['ROUGE-SU4-Precision']:.4f}  "
        f"R={generation['ROUGE-SU4-Recall']:.4f}  "
        f"F1={generation['ROUGE-SU4-F1']:.4f}"
    )
    print(
        f"  Citation: P={citation['Citation-Precision']:.4f}  "
        f"R={citation['Citation-Recall']:.4f}  "
        f"F1={citation['Citation-F1']:.4f}"
    )
    print("=" * 60)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build CLI parser for MedAESQA generation evaluation scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (for quick testing).",
    )
    return parser
