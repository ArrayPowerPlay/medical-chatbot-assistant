"""
BioASQ test retrieval evaluation entrypoint.

Outputs:
    results/test_results/bioasq/retrieval/detail.jsonl
    results/test_results/bioasq/retrieval/summary.json
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


TEST_PATH = settings.DATA_PATH / "test" / "test_bioasq.jsonl"
OUTPUT_DIR = settings.TEST_RESULTS_PATH / "bioasq" / "full_system"


def evaluate(limit: int | None = None) -> None:
    """Run retrieval evaluation on the BioASQ test split."""
    common.evaluate_split(
        data_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        split_name="test",
        limit=limit,
    )


def build_arg_parser():
    """Build CLI parser for test retrieval evaluation."""
    return common.build_arg_parser(
        "Evaluate the BioASQ retrieval pipeline on the test split."
    )


if __name__ == "__main__":
    setup_logging()
    from scripts.evaluation.shared.config_helper import load_and_apply_config
    load_and_apply_config("bioasq", "retrieval")
    
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate(limit=args.limit)
