"""
Shared helpers for BioASQ generation tuning scripts.
"""
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Configure project root
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings


STAGE1_LIMIT = 15
STAGE2_LIMIT = 25
TOP_K_STAGE1_SHORTLIST = 5


def build_baseline_config() -> Dict[str, Any]:
    """Build the default tuning baseline from current settings."""
    return {
        "generation_temperature": settings.GENERATION_TEMPERATURE,
        "generation_max_tokens": settings.GENERATION_MAX_TOKENS,
        "kg_top_k": settings.KG_TOP_K,
        "kg_hop1_m": settings.KG_HOP1_M,
        "kg_hop2_n": settings.KG_HOP2_N,
        "kg_hop2_cap": settings.KG_HOP2_CAP,
        "rerank_kg_top_n": settings.RERANK_KG_TOP_N,
        "use_kg_merger": settings.USE_KG_MERGER,
        "use_head_tail_placement": settings.USE_HEAD_TAIL_PLACEMENT,
    }


def build_stage1_search_space() -> Dict[str, List[Any]]:
    """Return the agreed Stage 1 one-factor-at-a-time search space."""
    return {
        "generation_temperature": [0.0],
        "generation_max_tokens": [256, 512],
        "kg_top_k": [2],
        "kg_hop1_m": [5],
        "kg_hop2_n": [3, 5, 8],
        "kg_hop2_cap": [30, 50, 80],
        "rerank_kg_top_n": [10, 20, 30],
        "use_kg_merger": [True],
        "use_head_tail_placement": [True],
    }


def generate_stage1_candidates() -> List[Dict[str, Any]]:
    """Generate baseline + one-factor-at-a-time candidate configs."""
    baseline = build_baseline_config()
    search_space = build_stage1_search_space()

    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add_candidate(config: Dict[str, Any]) -> None:
        canonical = json.dumps(config, sort_keys=True) # Sort the dictionary keys alphabetically
        if canonical in seen:
            return
        seen.add(canonical)
        candidates.append(config.copy())

    _add_candidate(baseline)
    for key, values in search_space.items():
        for value in values:
            config = baseline.copy()
            config[key] = value
            _add_candidate(config)

    return candidates


def candidate_run_name(index: int, config: Dict[str, Any]) -> str:
    """Build a readable run directory name for one candidate."""
    compact = (
        f"k{config['kg_top_k']}_"
        f"h1{config['kg_hop1_m']}_"
        f"h2{config['kg_hop2_n']}_"
        f"cap{config['kg_hop2_cap']}_"
        f"rkg{config['rerank_kg_top_n']}_"
        f"t{str(config['generation_temperature']).replace('.', 'p')}_"
        f"mt{config['generation_max_tokens']}_"
        f"kgm{int(config['use_kg_merger'])}_"
        f"ht{int(config['use_head_tail_placement'])}"
    )
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", compact).strip("_")
    return f"{index:03d}_{slug}"


def tuning_output_root() -> Path:
    """Return the root output directory for generation tuning results."""
    return settings.EVAL_RESULTS_PATH / "bioasq" / "generation" / "tuning"


def stage1_output_root() -> Path:
    """Return the Stage 1 tuning output directory."""
    return tuning_output_root() / "stage1"


def stage2_output_root() -> Path:
    """Return the Stage 2 tuning output directory."""
    return tuning_output_root() / "stage2"


def save_json(data: Any, path: Path) -> None:
    """Save one JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load one JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Load one generation evaluation summary.json file."""
    return load_json(summary_path)


def stage1_primary_score(summary: Dict[str, Any]) -> float:
    """Return the Stage 1 ranking score from one summary."""
    return float(summary["generation_metrics"].get("ROUGE-SU4-F1", 0.0))


def stage2_sort_key(summary: Dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the Stage 2 ranking tuple."""
    generation = summary.get("generation_metrics", {})
    ragas = summary.get("ragas_metrics", {})
    return (
        float(generation.get("ROUGE-SU4-F1", 0.0)),
        float(ragas.get("answer_correctness", 0.0)),
        float(ragas.get("faithfulness", 0.0)),
        float(ragas.get("answer_relevancy", 0.0)),
    )


def shortlist_top_k(records: Iterable[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Return the top-k records sorted by Stage 1 ROUGE-SU4-F1."""
    return sorted(
        records,
        key=lambda record: float(record.get("score", 0.0)),
        reverse=True,
    )[:top_k]
