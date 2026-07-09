"""
BioASQ test generation evaluation entrypoint for BASELINE VECTOR.
(Vector Search only, no BM25, no KG)

Outputs:
    results/eval_results/bioasq/baseline_vector/detail.jsonl
    results/eval_results/bioasq/baseline_vector/summary.json
    results/eval_results/bioasq/baseline_vector/predictions.jsonl
"""

import argparse
import sys
import asyncio
from pathlib import Path

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import setup_logging
from config.settings import settings
from scripts.evaluation.shared import generation_bioasq_common as common


TEST_PATH = settings.DATA_PATH / "test" / "test_bioasq.jsonl"
OUTPUT_DIR = project_root / "results" / "test_results" / "bioasq" / "baseline_vector"


def evaluate(
    limit: int | None = None,
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
    """Run generation evaluation on the BioASQ test split."""
    await common.evaluate_split(
        data_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        split_name="test",
        limit=limit,
        use_ragas=use_ragas,
        use_kg=False,       # ABLATION: KG is disabled
        use_vector=True,
        use_bm25=False,     # ABLATION: BM25 is disabled
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


def _str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for BioASQ test generation evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the BioASQ BASELINE VECTOR on the test split."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-ragas", type=_str_to_bool, default=True)
    parser.add_argument("--kg-top-k", type=int, default=settings.KG_TOP_K)
    parser.add_argument("--kg-hop1-m", type=int, default=settings.KG_HOP1_M)
    parser.add_argument("--kg-hop2-n", type=int, default=settings.KG_HOP2_N)
    parser.add_argument("--kg-hop2-cap", type=int, default=settings.KG_HOP2_CAP)
    parser.add_argument("--rerank-kg-top-n", type=int, default=settings.RERANK_KG_TOP_N)
    parser.add_argument("--generation-temperature", type=float, default=settings.GENERATION_TEMPERATURE)
    parser.add_argument("--use-kg-merger", type=_str_to_bool, default=settings.USE_KG_MERGER)
    parser.add_argument(
        "--use-head-tail-placement",
        type=_str_to_bool,
        default=settings.USE_HEAD_TAIL_PLACEMENT,
    )
    parser.add_argument("--generation-max-tokens", type=int, default=settings.GENERATION_MAX_TOKENS)
    return parser


if __name__ == "__main__":
    setup_logging()
    from scripts.evaluation.shared.config_helper import load_and_apply_config
    load_and_apply_config("bioasq", "generation")
    
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(evaluate(
        limit=args.limit,
        use_ragas=args.use_ragas,
        kg_top_k=args.kg_top_k,
        kg_hop1_m=args.kg_hop1_m,
        kg_hop2_n=args.kg_hop2_n,
        kg_hop2_cap=args.kg_hop2_cap,
        rerank_kg_top_n=args.rerank_kg_top_n,
        generation_temperature=args.generation_temperature,
        use_kg_merger=args.use_kg_merger,
        use_head_tail_placement=args.use_head_tail_placement,
        generation_max_tokens=args.generation_max_tokens,
    )
