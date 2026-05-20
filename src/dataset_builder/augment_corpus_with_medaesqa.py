"""Augment the main corpus with MedAESQA gold PMIDs using PubMed and PMC fallbacks."""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.dataset_builder.preprocess_bioasq_taskB import (
    fetch_pmc_fulltext_records,
    fetch_pubmed_records,
)


PMID_GROUP_PATTERN = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def extract_pmids_from_text(text: str) -> Set[str]:
    """Extract PMID groups formatted like [12345, 67890] from free text."""
    pmids: Set[str] = set()
    for group in PMID_GROUP_PATTERN.findall(text or ""):
        for pmid in group.split(","):
            cleaned = pmid.strip()
            if cleaned:
                pmids.add(cleaned)
    return pmids


def collect_gold_pmids(dataset: List[Dict]) -> Set[str]:
    """Collect PMIDs only from MedAESQA expert-curated answers."""
    pmids: Set[str] = set()
    for item in dataset:
        pmids.update(extract_pmids_from_text(item.get("expert_curated_answer", "")))
    return pmids


def get_existing_pmids(corpus_path: Path) -> Set[str]:
    """Load PMIDs already present in corpus.jsonl."""
    pmids: Set[str] = set()
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pmid = str(data.get("pmid", "")).strip()
                if pmid:
                    pmids.add(pmid)
    return pmids


def append_articles(corpus_path: Path, articles: Iterable[Dict]) -> List[str]:
    """Append new corpus articles and return the appended PMID list."""
    appended_pmids: List[str] = []
    seen_pmids: Set[str] = set()
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "a", encoding="utf-8") as f:
        for article in articles:
            pmid = str(article.get("pmid", "")).strip()
            if not pmid or pmid in seen_pmids:
                continue

            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            appended_pmids.append(pmid)
            seen_pmids.add(pmid)

    return appended_pmids


def _load_medaesqa_dataset(medaesqa_path: Path) -> List[Dict]:
    if not medaesqa_path.exists():
        raise FileNotFoundError(f"MedAESQA file not found: {medaesqa_path}")

    with open(medaesqa_path, "r", encoding="utf-8-sig") as f:
        medaesqa_data = json.load(f)

    if not isinstance(medaesqa_data, list):
        raise ValueError("Expected MedAESQA JSON top-level value to be a list.")

    return medaesqa_data


def _build_manifest(
    medaesqa_path: Path,
    corpus_path: Path,
    target_pmids: Set[str],
    existing_pmids: Set[str],
    appended_articles: List[Dict],
    appended_pubmed_pmids: List[str],
    appended_pmc_pmids: List[str],
    unresolved_no_abstract_no_pmc: List[str],
    unresolved_fetch_or_parse_failure: Dict[str, str],
) -> Dict:
    """Build a structured manifest for one MedAESQA corpus augmentation run."""
    appended_pmids = [str(article["pmid"]) for article in appended_articles]
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "medaesqa_path": str(medaesqa_path),
        "corpus_path": str(corpus_path),
        "stats": {
            "gold_pmids_total": len(target_pmids),
            "already_in_corpus": len(target_pmids & existing_pmids),
            "fetched_from_pubmed": len(appended_pubmed_pmids),
            "fetched_from_pmc_fallback": len(appended_pmc_pmids),
            "unresolved_no_abstract_no_pmc": len(unresolved_no_abstract_no_pmc),
            "unresolved_fetch_or_parse_failure": len(unresolved_fetch_or_parse_failure),
            "appended_total": len(appended_pmids),
        },
        "appended_pmids": appended_pmids,
        "appended_from_pubmed": appended_pubmed_pmids,
        "appended_from_pmc_fallback": appended_pmc_pmids,
        "unresolved_no_abstract_no_pmc": sorted(unresolved_no_abstract_no_pmc),
        "unresolved_fetch_or_parse_failure": [
            {"pmid": pmid, "reason": reason}
            for pmid, reason in sorted(unresolved_fetch_or_parse_failure.items())
        ],
    }


def augment_corpus_with_medaesqa_pmids(
    medaesqa_path: Path = Path("data/raw/medaesqa_v1.json"),
    corpus_path: Path = Path("data/corpus/corpus.jsonl"),
    manifest_path: Path = Path("results/eval_results/medaesqa/corpus_augment_manifest.json"),
) -> Dict:
    """
    Augment corpus.jsonl with MedAESQA gold PMIDs only.

    Strategy:
        1. Collect PMIDs only from expert_curated_answer.
        2. Fetch PubMed metadata with robust parsing and single-PMID retry.
        3. For records with no PubMed abstract, fallback to PMC full text when PMCID exists.
        4. Persist a manifest for reproducibility.
    """
    medaesqa_data = _load_medaesqa_dataset(medaesqa_path)
    target_pmids = collect_gold_pmids(medaesqa_data)
    existing_pmids = get_existing_pmids(corpus_path)
    missing_pmids = sorted(target_pmids - existing_pmids)

    print(f"MedAESQA questions: {len(medaesqa_data)}")
    print(f"Gold PMIDs referenced in MedAESQA: {len(target_pmids)}")
    print(f"Existing gold PMIDs already in corpus: {len(target_pmids & existing_pmids)}")
    print(f"Missing gold PMIDs to fetch: {len(missing_pmids)}")

    appended_articles: List[Dict] = []
    appended_pubmed_pmids: List[str] = []
    appended_pmc_pmids: List[str] = []
    unresolved_no_abstract_no_pmc: List[str] = []
    unresolved_fetch_or_parse_failure: Dict[str, str] = {}

    if missing_pmids:
        pubmed_records, pubmed_statuses = fetch_pubmed_records(missing_pmids)
        pmc_candidates: Dict[str, str] = {}

        for pmid in missing_pmids:
            status_info = pubmed_statuses.get(pmid, {"status": "parse_failed"})
            status = status_info.get("status", "parse_failed")
            record = pubmed_records.get(pmid)

            if status == "fetched_pubmed_abstract" and record:
                article = {
                    "pmid": pmid,
                    "title": record.get("title", ""),
                    "abstractText": record.get("abstractText", ""),
                    "content_source": "pubmed_abstract",
                }
                if record.get("pmcid"):
                    article["pmcid"] = record["pmcid"]
                appended_articles.append(article)
                appended_pubmed_pmids.append(pmid)
                continue

            if status == "no_pubmed_abstract" and record:
                pmcid = status_info.get("pmcid") or record.get("pmcid", "")
                if pmcid:
                    pmc_candidates[pmid] = pmcid
                else:
                    unresolved_no_abstract_no_pmc.append(pmid)
                continue

            unresolved_fetch_or_parse_failure[pmid] = status

        if pmc_candidates:
            pmc_records, pmc_failures = fetch_pmc_fulltext_records(pmc_candidates)
            for pmid, record in pmc_records.items():
                article = {
                    "pmid": pmid,
                    "title": record.get("title", "") or pubmed_records.get(pmid, {}).get("title", ""),
                    "abstractText": record.get("abstractText", ""),
                    "content_source": "pmc_fulltext_fallback",
                    "pmcid": record.get("pmcid", pmc_candidates.get(pmid, "")),
                }
                appended_articles.append(article)
                appended_pmc_pmids.append(pmid)

            for pmid, reason in pmc_failures.items():
                if reason == "no_pmcid":
                    unresolved_no_abstract_no_pmc.append(pmid)
                else:
                    unresolved_fetch_or_parse_failure[pmid] = reason

    appended_articles = sorted(appended_articles, key=lambda article: str(article["pmid"]))
    appended_pmids = append_articles(corpus_path, appended_articles)

    manifest = _build_manifest(
        medaesqa_path=medaesqa_path,
        corpus_path=corpus_path,
        target_pmids=target_pmids,
        existing_pmids=existing_pmids,
        appended_articles=appended_articles,
        appended_pubmed_pmids=sorted(appended_pubmed_pmids),
        appended_pmc_pmids=sorted(appended_pmc_pmids),
        unresolved_no_abstract_no_pmc=sorted(set(unresolved_no_abstract_no_pmc)),
        unresolved_fetch_or_parse_failure=unresolved_fetch_or_parse_failure,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Appended {len(appended_pmids)} new gold articles to {corpus_path}")
    print(f"  - from PubMed abstract: {len(appended_pubmed_pmids)}")
    print(f"  - from PMC fallback: {len(appended_pmc_pmids)}")
    print(f"  - unresolved (no abstract + no PMC): {len(set(unresolved_no_abstract_no_pmc))}")
    print(f"  - unresolved (fetch/parse failure): {len(unresolved_fetch_or_parse_failure)}")
    print(f"Manifest saved to {manifest_path}")

    return manifest


if __name__ == "__main__":
    augment_corpus_with_medaesqa_pmids()
