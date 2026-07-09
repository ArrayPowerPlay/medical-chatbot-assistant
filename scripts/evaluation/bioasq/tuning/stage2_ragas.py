"""
Stage 2 BioASQ generation tuning:
- run ROUGE-SU4 + full RAGAS
- use the first 50 validation questions
- evaluate the top-5 configs from Stage 1
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import setup_logging
from config.settings import settings
from scripts.evaluation.shared import generation_bioasq_common as common
from scripts.evaluation.bioasq.tuning import tuning_common


VAL_PATH = settings.DATA_PATH / "val" / "val_bioasq.jsonl"


async def run_stage2(
    limit: int = tuning_common.STAGE2_LIMIT,
    shortlist_path: Path | None = None,
) -> None:
    """Run Stage 2 tuning on the Stage 1 top-5 shortlist."""
    output_root = tuning_common.stage2_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    if shortlist_path is None:
        shortlist_path = tuning_common.stage1_output_root() / "top5_candidates.json"

    shortlist = tuning_common.load_json(shortlist_path)
    leaderboard: List[dict] = []

    for index, candidate in enumerate(shortlist, 1):
        config = candidate["config"]
        run_name = tuning_common.candidate_run_name(index, config)
        run_output_dir = output_root / run_name

        await common.evaluate_split(
            data_path=VAL_PATH,
            output_dir=run_output_dir,
            split_name="validation",
            limit=limit,
            use_ragas=True,
            kg_top_k=config["kg_top_k"],
            kg_hop1_m=config["kg_hop1_m"],
            kg_hop2_n=config["kg_hop2_n"],
            kg_hop2_cap=config["kg_hop2_cap"],
            rerank_kg_top_n=config["rerank_kg_top_n"],
            generation_temperature=config["generation_temperature"],
            use_kg_merger=config["use_kg_merger"],
            use_head_tail_placement=config["use_head_tail_placement"],
            generation_max_tokens=config["generation_max_tokens"],
        )

        summary_path = run_output_dir / "summary.json"
        summary = tuning_common.load_summary(summary_path)
        leaderboard.append(
            {
                "rank_candidate": run_name,
                "summary_path": str(summary_path),
                "config": summary["config"],
                "generation_metrics": summary["generation_metrics"],
                "ragas_metrics": summary["ragas_metrics"],
                "stage2_sort_key": list(tuning_common.stage2_sort_key(summary)),
            }
        )

    leaderboard = sorted(
        leaderboard,
        key=lambda item: tuple(item["stage2_sort_key"]),
        reverse=True,
    )
    tuning_common.save_json(leaderboard, output_root / "leaderboard.json")
    if leaderboard:
        tuning_common.save_json(leaderboard[0], output_root / "best_config.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 BioASQ generation tuning.")
    parser.add_argument("--limit", type=int, default=tuning_common.STAGE2_LIMIT)
    parser.add_argument(
        "--shortlist-path",
        type=Path,
        default=tuning_common.stage1_output_root() / "top5_candidates.json",
    )
    return parser


if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser().parse_args()
    import asyncio
    asyncio.run(run_stage2(limit=args.limit, shortlist_path=args.shortlist_path))
