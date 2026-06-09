"""
MedAESQA test generation evaluation entrypoint.

Metrics: ROUGE-SU4-F1, Citation Precision / Recall / F1.
RAGAS is intentionally disabled for MedAESQA (secondary dataset).
use_citations defaults to True — citation quality IS the primary signal here.

Outputs:
    results/test_results/medaesqa/generation/detail.jsonl
    results/test_results/medaesqa/generation/summary.json
    results/test_results/medaesqa/generation/predictions.jsonl
"""

import argparse
import sys
from pathlib import Path

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import setup_logging
from config.settings import settings
from scripts.evaluation.shared import generation_medaesqa_common as common

TEST_PATH = settings.DATA_PATH / "test" / "test_medaesqa.jsonl"
OUTPUT_DIR = settings.TEST_RESULTS_PATH / "medaesqa" / "generation"


def evaluate(
    limit: int | None = None,
    kg_top_k: int = settings.KG_TOP_K,
    kg_hop1_m: int = settings.KG_HOP1_M,
    kg_hop2_n: int = settings.KG_HOP2_N,
    kg_hop2_cap: int = settings.KG_HOP2_CAP,
    rerank_kg_top_n: int = settings.RERANK_KG_TOP_N,
    generation_temperature: float = settings.GENERATION_TEMPERATURE,
    use_kg_merger: bool = settings.USE_KG_MERGER,
    use_head_tail_placement: bool = settings.USE_HEAD_TAIL_PLACEMENT,
    generation_max_tokens: int = settings.GENERATION_MAX_TOKENS,
    use_citations: bool = True,  # Always on for MedAESQA citation benchmark
) -> None:
    """Run generation evaluation on the MedAESQA test split."""
    common.evaluate_split(
        data_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        split_name="test",
        limit=limit,
        kg_top_k=kg_top_k,
        kg_hop1_m=kg_hop1_m,
        kg_hop2_n=kg_hop2_n,
        kg_hop2_cap=kg_hop2_cap,
        rerank_kg_top_n=rerank_kg_top_n,
        generation_temperature=generation_temperature,
        use_kg_merger=use_kg_merger,
        use_head_tail_placement=use_head_tail_placement,
        generation_max_tokens=generation_max_tokens,
        use_citations=use_citations,
    )


def _str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for MedAESQA test generation evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the MedAESQA generation pipeline on the test split."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--kg-top-k", type=int, default=settings.KG_TOP_K)
    parser.add_argument("--kg-hop1-m", type=int, default=settings.KG_HOP1_M)
    parser.add_argument("--kg-hop2-n", type=int, default=settings.KG_HOP2_N)
    parser.add_argument("--kg-hop2-cap", type=int, default=settings.KG_HOP2_CAP)
    parser.add_argument("--rerank-kg-top-n", type=int, default=settings.RERANK_KG_TOP_N)
    parser.add_argument(
        "--generation-temperature", type=float, default=settings.GENERATION_TEMPERATURE
    )
    parser.add_argument("--use-kg-merger", type=_str_to_bool, default=settings.USE_KG_MERGER)
    parser.add_argument(
        "--use-head-tail-placement",
        type=_str_to_bool,
        default=settings.USE_HEAD_TAIL_PLACEMENT,
    )
    parser.add_argument(
        "--generation-max-tokens", type=int, default=settings.GENERATION_MAX_TOKENS
    )
    parser.add_argument(
        "--use-citations",
        type=_str_to_bool,
        default=True,
        help="Cite PMIDs in generated answers (default: True for citation benchmark).",
    )
    return parser


if __name__ == "__main__":
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate(
        limit=args.limit,
        kg_top_k=args.kg_top_k,
        kg_hop1_m=args.kg_hop1_m,
        kg_hop2_n=args.kg_hop2_n,
        kg_hop2_cap=args.kg_hop2_cap,
        rerank_kg_top_n=args.rerank_kg_top_n,
        generation_temperature=args.generation_temperature,
        use_kg_merger=args.use_kg_merger,
        use_head_tail_placement=args.use_head_tail_placement,
        generation_max_tokens=args.generation_max_tokens,
        use_citations=args.use_citations,
    )
