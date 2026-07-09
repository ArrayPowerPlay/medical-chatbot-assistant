"""
BioASQ validation retrieval evaluation entrypoint.

Outputs:
    results/eval_results/bioasq/retrieval/detail.jsonl
    results/eval_results/bioasq/retrieval/summary.json
"""

import sys
from pathlib import Path

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import setup_logging
from config.settings import settings
from scripts.evaluation.shared import retrieval_common as common


VAL_PATH = settings.DATA_PATH / "val" / "val_bioasq.jsonl"
OUTPUT_DIR = project_root / "results" / "val_results" / "bioasq" / "full_system"


async def evaluate(limit: int | None = None) -> None:
    """Run retrieval evaluation on the BioASQ validation split."""
    await common.evaluate_split(
        data_path=VAL_PATH,
        output_dir=OUTPUT_DIR,
        split_name="validation",
        limit=limit,
    )


def build_arg_parser():
    """Build CLI parser for retrieval evaluation."""
    return common.build_arg_parser(
        "Evaluate the BioASQ retrieval pipeline on the validation split."
    )


if __name__ == "__main__":
    setup_logging()
    from scripts.evaluation.shared.config_helper import load_and_apply_config
    load_and_apply_config("bioasq", "retrieval")
    
    parser = build_arg_parser()
    args = parser.parse_args()
    import asyncio
    asyncio.run(evaluate(limit=args.limit))
