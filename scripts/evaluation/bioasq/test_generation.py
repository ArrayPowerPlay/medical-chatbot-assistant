"""
BioASQ test generation evaluation entrypoint.

Future outputs should be written under:
    results/test_results/bioasq/generation/
"""

import sys
from pathlib import Path

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings


TEST_PATH = settings.DATA_PATH / "test" / "test_bioasq.jsonl"
OUTPUT_DIR = settings.TEST_RESULTS_PATH / "bioasq" / "generation"
