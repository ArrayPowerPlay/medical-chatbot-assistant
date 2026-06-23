"""
MedAESQA generation tuning script (One-Factor-At-A-Time).

Evaluates generation parameters by varying one parameter at a time from a strong
baseline on the first 15 questions of the MedAESQA test set. Enables measuring the
impact of token limits, temperature, KG sub-graph settings, and context layout.

Usage:
    python scripts/evaluation/medaesqa/tuning/tune_generation.py --limit 15
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import logger, setup_logging
from config.settings import settings
from scripts.evaluation.shared import generation_medaesqa_common as common

TEST_PATH = settings.DATA_PATH / "test" / "test_medaesqa.jsonl"
OUTPUT_DIR = settings.EVAL_RESULTS_PATH / "medaesqa" / "generation"

# Define the baseline parameter set
BASELINE: Dict[str, Any] = {
    "generation_max_tokens": 2048,
    "generation_temperature": 0.0,
    "use_kg_merger": True,
    "use_head_tail_placement": True,
    "kg_top_k": 2,
    "kg_hop1_m": 3,
    "kg_hop2_n": 3,
    "kg_hop2_cap": 30,
    "rerank_kg_top_n": 10,
}

# Parameters to vary one-factor-at-a-time (OFAT)
CANDIDATE_VARY: Dict[str, List[Any]] = {
    "generation_max_tokens": [1024],
    "kg_top_k": [3],  
    "kg_hop1_m": [5],
    "kg_hop2_n": [5, 7],
    "kg_hop2_cap": [50, 80],
    "rerank_kg_top_n": [20, 30],
}


def run_tuning(limit: int) -> None:
    """Run generation evaluation across OFAT candidates and summarize results."""
    logger.info(f"Starting MedAESQA generation OFAT tuning on first {limit} questions...")
    
    # Construct the list of configurations to test
    configs: List[Dict[str, Any]] = []
    
    # 1. Baseline
    configs.append({
        "name": "Baseline",
        "params": BASELINE.copy()
    })
    
    # 2. OFAT Variations
    for param, values in CANDIDATE_VARY.items():
        for val in values:
            cfg_params = BASELINE.copy()
            cfg_params[param] = val
            configs.append({
                "name": f"{param}={val}",
                "params": cfg_params
            })
            
    results: List[Dict[str, Any]] = []
    
    for idx, cfg in enumerate(configs, 1):
        name = cfg["name"]
        p = cfg["params"]
        
        # Use a safe output directory name for each run (replace special characters)
        safe_name = name.replace("=", "_").replace(".", "_")
        run_output_dir = OUTPUT_DIR / f"run_ofat_{safe_name}"
        
        logger.info(f"\n============================================================")
        logger.info(f"RUNNING GENERATION RUN {idx}/{len(configs)}: {name}")
        logger.info(f"============================================================")
        
        try:
            # Run the standard evaluation split with current parameters
            common.evaluate_split(
                data_path=TEST_PATH,
                output_dir=run_output_dir,
                split_name=f"tune_ofat_{limit}_{safe_name}",
                limit=limit,
                kg_top_k=p["kg_top_k"],
                kg_hop1_m=p["kg_hop1_m"],
                kg_hop2_n=p["kg_hop2_n"],
                kg_hop2_cap=p["kg_hop2_cap"],
                rerank_kg_top_n=p["rerank_kg_top_n"],
                generation_temperature=p["generation_temperature"],
                use_kg_merger=p["use_kg_merger"],
                use_head_tail_placement=p["use_head_tail_placement"],
                generation_max_tokens=p["generation_max_tokens"],
                use_citations=True,
            )
            
            # Read the summary file back
            summary_path = run_output_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                
                results.append({
                    "config_name": name,
                    "params": p,
                    "rouge": summary_data["generation_metrics"],
                    "citation": summary_data["citation_metrics"]
                })
            else:
                logger.error(f"Summary file not found at {summary_path}")
                
        except Exception:
            logger.exception(f"Failed generation run for configuration {name}")
            
    # Print comparative results table
    print("\n" + "=" * 95)
    print(f"MEDAESQA GENERATION TUNING COMPARISON (OFAT, Limit={limit} questions)")
    print("=" * 95)
    print(f"{'Configuration':<30} | {'ROUGE-P':<8} {'ROUGE-R':<8} {'ROUGE-F1':<8} | {'Cite-P':<8} {'Cite-R':<8} {'Cite-F1':<8}")
    print("-" * 95)
    for res in results:
        name = res["config_name"]
        r = res["rouge"]
        c = res["citation"]
        print(
            f"{name:<30} | "
            f"{r.get('ROUGE-SU4-Precision', 0.0):.4f} "
            f"{r.get('ROUGE-SU4-Recall', 0.0):.4f} "
            f"{r.get('ROUGE-SU4-F1', 0.0):.4f} | "
            f"{c.get('Citation-Precision', 0.0):.4f} "
            f"{c.get('Citation-Recall', 0.0):.4f} "
            f"{c.get('Citation-F1', 0.0):.4f}"
        )
    print("=" * 95)
    
    # Save combined tuning summary
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    combined_summary_path = OUTPUT_DIR / f"tune_summary_ofat_{timestamp}.json"
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "limit": limit,
            "results": results
        }, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Combined OFAT tuning summary saved to {combined_summary_path}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Tune MedAESQA generation parameters using OFAT.")
    parser.add_argument("--limit", type=int, default=15, help="Number of questions to evaluate (default: 15).")
    args = parser.parse_args()

    run_tuning(args.limit)
