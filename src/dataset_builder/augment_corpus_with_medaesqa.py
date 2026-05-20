import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.dataset_builder.preprocess_bioasq_taskA import get_existing_pmids
from src.dataset_builder.preprocess_bioasq_taskB import fetch_pubmed_data


PMID_GROUP_PATTERN = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def extract_pmids_from_text(text: str) -> Set[str]:
    pmids: Set[str] = set()
    for group in PMID_GROUP_PATTERN.findall(text or ""):
        for pmid in group.split(","):
            cleaned = pmid.strip()
            if cleaned:
                pmids.add(cleaned)
    return pmids


def collect_medaesqa_pmids(dataset: List[Dict]) -> Set[str]:
    """Collect PMIDs from MedAESQA gold answers and annotated citation assessments."""
    pmids: Set[str] = set()

    for item in dataset:
        pmids.update(extract_pmids_from_text(item.get("expert_curated_answer", "")))

        for answer in item.get("machine_generated_answers", {}).values():
            for sentence in answer.get("answer_sentences", []):
                for citation in sentence.get("citation_assessment") or []:
                    cited_pmid = str(citation.get("cited_pmid", "")).strip()
                    if cited_pmid:
                        pmids.add(cited_pmid)

    return pmids


def append_articles(corpus_path: Path, articles: Iterable[Dict]) -> int:
    count = 0
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "a", encoding="utf-8") as f:
        for article in articles:
            if not article.get("pmid"):
                continue
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            count += 1

    return count


def augment_corpus_with_medaesqa_pmids(
    medaesqa_path: Path = Path("data/raw/medaesqa_v1.json"),
    corpus_path: Path = Path("data/corpus/corpus.jsonl"),
) -> None:
    if not medaesqa_path.exists():
        raise FileNotFoundError(f"MedAESQA file not found: {medaesqa_path}")

    with open(medaesqa_path, "r", encoding="utf-8") as f:
        medaesqa_data = json.load(f)

    if not isinstance(medaesqa_data, list):
        raise ValueError("Expected MedAESQA JSON top-level value to be a list.")

    target_pmids = collect_medaesqa_pmids(medaesqa_data)
    existing_pmids = get_existing_pmids(corpus_path)
    missing_pmids = sorted(target_pmids - existing_pmids)

    print(f"MedAESQA questions: {len(medaesqa_data)}")
    print(f"Unique PMIDs referenced in MedAESQA: {len(target_pmids)}")
    print(f"Existing PMIDs already in corpus: {len(target_pmids & existing_pmids)}")
    print(f"Missing PMIDs to fetch: {len(missing_pmids)}")

    if not missing_pmids:
        print("Corpus already covers all MedAESQA PMIDs. Nothing to append.")
        return

    fetched_articles = fetch_pubmed_data(missing_pmids)
    fetched_valid = [
        article
        for pmid, article in fetched_articles.items()
        if pmid not in existing_pmids and article.get("title") and article.get("abstractText")
    ]

    appended_count = append_articles(corpus_path, fetched_valid)
    fetched_valid_pmids = {article["pmid"] for article in fetched_valid}
    unresolved_pmids = sorted(set(missing_pmids) - fetched_valid_pmids)

    print(f"Appended {appended_count} new articles to {corpus_path}")
    if unresolved_pmids:
        print(f"Could not append {len(unresolved_pmids)} PMIDs (missing title/abstract or fetch failure).")
        print(", ".join(unresolved_pmids))


if __name__ == "__main__":
    augment_corpus_with_medaesqa_pmids()
