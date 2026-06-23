"""
BioASQ validation generation evaluation entrypoint.

For generation tuning, use --generation-temperature to sweep values:
    python -m scripts.evaluation.bioasq.val_generation --generation-temperature 0.0
    python -m scripts.evaluation.bioasq.val_generation --generation-temperature 0.1

Outputs:
    results/eval_results/bioasq/generation/detail.jsonl
    results/eval_results/bioasq/generation/summary.json
    results/eval_results/bioasq/generation/predictions.jsonl
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
from scripts.evaluation.shared import generation_bioasq_common as common


VAL_PATH = settings.DATA_PATH / "val" / "val_bioasq.jsonl"
OUTPUT_DIR = settings.EVAL_RESULTS_PATH / "bioasq" / "generation"


def evaluate(
    limit: int | None = None,
    use_ragas: bool = True,
    generation_temperature: float = settings.GENERATION_TEMPERATURE,
    kg_top_k: int = settings.KG_TOP_K,
    kg_hop1_m: int = settings.KG_HOP1_M,
    kg_hop2_n: int = settings.KG_HOP2_N,
    kg_hop2_cap: int = settings.KG_HOP2_CAP,
    rerank_kg_top_n: int = settings.RERANK_KG_TOP_N,
    use_kg_merger: bool = settings.USE_KG_MERGER,
    use_head_tail_placement: bool = settings.USE_HEAD_TAIL_PLACEMENT,
    generation_max_tokens: int = settings.GENERATION_MAX_TOKENS,
) -> None:
    """Run generation evaluation on the BioASQ validation split."""
    common.evaluate_split(
        data_path=VAL_PATH,
        output_dir=OUTPUT_DIR,
        split_name="validation",
        limit=limit,
        use_ragas=use_ragas,
        generation_temperature=generation_temperature,
        kg_top_k=kg_top_k,
        kg_hop1_m=kg_hop1_m,
        kg_hop2_n=kg_hop2_n,
        kg_hop2_cap=kg_hop2_cap,
        rerank_kg_top_n=rerank_kg_top_n,
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
    """Build CLI parser for validation generation evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the BioASQ generation pipeline on the validation split."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--use-ragas",
        type=_str_to_bool,
        default=False,
        help="Run RAGAS metrics (requires OPENAI_API_KEY). Default: False for faster val runs.",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=settings.GENERATION_TEMPERATURE,
        help="LLM sampling temperature. Sweep 0.0/0.1/0.2/0.3 to tune ROUGE-SU4.",
    )
    parser.add_argument("--kg-top-k", type=int, default=settings.KG_TOP_K)
    parser.add_argument("--kg-hop1-m", type=int, default=settings.KG_HOP1_M)
    parser.add_argument("--kg-hop2-n", type=int, default=settings.KG_HOP2_N)
    parser.add_argument("--kg-hop2-cap", type=int, default=settings.KG_HOP2_CAP)
    parser.add_argument("--rerank-kg-top-n", type=int, default=settings.RERANK_KG_TOP_N)
    parser.add_argument("--use-kg-merger", type=_str_to_bool, default=settings.USE_KG_MERGER)
    parser.add_argument(
        "--use-head-tail-placement",
        type=_str_to_bool,
        default=settings.USE_HEAD_TAIL_PLACEMENT,
    )
    parser.add_argument(
        "--generation-max-tokens", type=int, default=settings.GENERATION_MAX_TOKENS
    )
    return parser


if __name__ == "__main__":
    setup_logging()
    from scripts.evaluation.shared.config_helper import load_and_apply_config
    load_and_apply_config("bioasq", "generation")
    
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate(
        limit=args.limit,
        use_ragas=args.use_ragas,
        generation_temperature=args.generation_temperature,
        kg_top_k=args.kg_top_k,
        kg_hop1_m=args.kg_hop1_m,
        kg_hop2_n=args.kg_hop2_n,
        kg_hop2_cap=args.kg_hop2_cap,
        rerank_kg_top_n=args.rerank_kg_top_n,
        use_kg_merger=args.use_kg_merger,
        use_head_tail_placement=args.use_head_tail_placement,
        generation_max_tokens=args.generation_max_tokens,
    )
