"""Build BioASQ validation/test QA files and shared PubMed/PMC fetch helpers."""

import json
import re
import sys
import time
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Tuple
import httpx
import xml.etree.ElementTree as ET

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _has_full_snippet_coverage(sample: Dict) -> bool:
    """Check if every relevant PMID in a question has at least one corresponding snippet.

    Args:
        sample: A single QA sample dict with 'relevant_pmid' and 'snippets' fields.

    Returns:
        True if all relevant PMIDs are covered by at least one snippet.
    """
    snippet_pmids = {s["pmid"] for s in sample.get("snippets", [])}
    relevant_pmids = set(sample.get("relevant_pmid", []))
    return relevant_pmids.issubset(snippet_pmids)


def setup_directories():
    """Create data folder structure"""
    dirs = {
        "raw": Path("data/raw"),
        "val": Path("data/val"),
        "test": Path("data/test"),
        "corpus": Path("data/corpus")
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def extract_pmid(url: str) -> str:
    """Extract PMID from PubMed URL"""
    if not url: return ""
    match = re.search(r'(\d+)$', url.strip())
    return match.group(1) if match else ""


def _normalize_xml_text(text: str) -> str:
    """Collapse internal whitespace extracted from XML nodes."""
    return re.sub(r"\s+", " ", text or "").strip()


def _element_text(element: ET.Element | None) -> str:
    """Extract recursively nested text from an XML element."""
    if element is None:
        return ""
    return _normalize_xml_text("".join(element.itertext()))


def _parse_pubmed_xml(xml_content: str) -> Dict[str, Dict]:
    """Parse the XML response from PubMed and extract article details."""
    results = {}
    try:
        root = ET.fromstring(xml_content)
        for article in root:
            if article.tag not in {"PubmedArticle", "PubmedBookArticle"}:
                continue

            pmid_element = article.find('.//PMID')
            pmid = _element_text(pmid_element)
            if not pmid:
                continue

            title_element = article.find('.//ArticleTitle')
            title = _element_text(title_element)

            abstract_texts = []
            for abstract_element in article.findall('.//AbstractText'):
                abstract_text = _element_text(abstract_element)
                if not abstract_text:
                    continue

                label = _normalize_xml_text(abstract_element.attrib.get("Label", ""))
                if label:
                    abstract_text = f"{label}: {abstract_text}"
                abstract_texts.append(abstract_text)
            abstract = _normalize_xml_text(" ".join(abstract_texts))

            pmcid_element = article.find(".//ArticleId[@IdType='pmc']")
            pmcid = _element_text(pmcid_element)
            if pmcid and not pmcid.upper().startswith("PMC"):
                pmcid = f"PMC{pmcid}"

            results[pmid] = {
                "pmid": pmid,
                "title": title.strip(),
                "abstractText": abstract.strip(),
                "pmcid": pmcid,
                "content_source": "pubmed_abstract" if abstract.strip() else "pubmed_no_abstract",
                "record_type": article.tag,
            }
    except ET.ParseError as e:
        print(f"\nError parsing XML: {e}")
    return results


def _request_with_retries(
    client: httpx.Client,
    url: str,
    params: Dict[str, str],
    request_label: str,
    max_retries: int = 3,
) -> httpx.Response | None:
    """Fetch one E-Utilities resource with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.RequestError as e:
            print(
                f"\nError fetching {request_label}. Attempt {attempt + 1}/{max_retries}. "
                f"Error: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except httpx.HTTPStatusError as e:
            print(
                f"\nHTTP error fetching {request_label}. Attempt {attempt + 1}/{max_retries}. "
                f"Error: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _classify_pubmed_record(record: Dict[str, str]) -> str:
    """Map a parsed PubMed record to a fetch status label."""
    if record.get("title") and record.get("abstractText"):
        return "fetched_pubmed_abstract"
    if record.get("title"):
        return "no_pubmed_abstract"
    return "parse_failed"


def fetch_pubmed_records(pmids: List[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, str]]]:
    """Fetch PubMed metadata and return both records and per-PMID status."""
    unique_pmids = list(dict.fromkeys(str(pmid).strip() for pmid in pmids if str(pmid).strip()))
    if not unique_pmids:
        return {}, {}

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    batch_size = 150
    total_pmids = len(unique_pmids)
    records: Dict[str, Dict] = {}
    statuses: Dict[str, Dict[str, str]] = {}

    print(f"Starting to fetch abstracts for {total_pmids} unique PMIDs...")

    with httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers=DEFAULT_HTTP_HEADERS,
    ) as client:
        for i in range(0, total_pmids, batch_size):
            batch = unique_pmids[i:i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }
            response = _request_with_retries(
                client=client,
                url=base_url,
                params=params,
                request_label=f"PubMed batch starting at index {i}",
            )
            if response is None:
                for pmid in batch:
                    statuses[pmid] = {"status": "fetch_failed"}
                continue

            batch_results = _parse_pubmed_xml(response.text)
            records.update(batch_results)

            for pmid, record in batch_results.items():
                statuses[pmid] = {
                    "status": _classify_pubmed_record(record),
                    "pmcid": record.get("pmcid", ""),
                }

            missing_in_batch = sorted(set(batch) - set(batch_results.keys()))
            if missing_in_batch:
                print(f"\nWarning: Missing PMIDs in batch parse. Retrying individually: {missing_in_batch}")

            for pmid in missing_in_batch:
                single_params = {
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "xml",
                    "rettype": "abstract",
                }
                single_response = _request_with_retries(
                    client=client,
                    url=base_url,
                    params=single_params,
                    request_label=f"PubMed PMID {pmid}",
                )
                if single_response is None:
                    statuses[pmid] = {"status": "fetch_failed"}
                    continue

                single_result = _parse_pubmed_xml(single_response.text)
                if pmid not in single_result:
                    statuses[pmid] = {"status": "parse_failed"}
                    continue

                record = single_result[pmid]
                records[pmid] = record
                statuses[pmid] = {
                    "status": _classify_pubmed_record(record),
                    "pmcid": record.get("pmcid", ""),
                }

            print(f"Fetched {len(records)}/{total_pmids} articles...", end="\r")
            time.sleep(0.4)

    for pmid in unique_pmids:
        statuses.setdefault(pmid, {"status": "parse_failed"})

    return records, statuses


def fetch_pubmed_data(pmids: List[str]) -> Dict[str, Dict]:
    """Backward-compatible wrapper returning only parsed PubMed records."""
    records, _ = fetch_pubmed_records(pmids)
    return records


def _parse_pmc_xml(xml_content: str, pmid: str, pmcid: str) -> Dict[str, str] | None:
    """Parse PMC XML and extract a plain-text fallback document."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return None

    title = ""
    title_element = root.find(".//front//article-title")
    if title_element is not None:
        title = _element_text(title_element)

    abstract_parts = []
    for abstract_element in root.findall(".//front//abstract"):
        abstract_text = _element_text(abstract_element)
        if abstract_text:
            abstract_parts.append(abstract_text)

    body_parts = []
    for paragraph in root.findall(".//body//p"):
        paragraph_text = _element_text(paragraph)
        if paragraph_text:
            body_parts.append(paragraph_text)

    full_text = _normalize_xml_text(" ".join(abstract_parts + body_parts))
    if not full_text:
        return None

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "title": title,
        "abstractText": full_text,
        "content_source": "pmc_fulltext_fallback",
    }


class _PMCBodyHTMLParser(HTMLParser):
    """Extract main article-body paragraph text from a PMC HTML page."""

    def __init__(self):
        super().__init__()
        self.in_body_section = False
        self.body_section_depth = 0
        self.capture_text = False
        self.capture_title = False
        self.current_text_parts: List[str] = []
        self.paragraphs: List[str] = []
        self.title_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        attrs_dict = dict(attrs)
        class_value = attrs_dict.get("class", "")

        if tag == "section" and "main-article-body" in class_value:
            self.in_body_section = True
            self.body_section_depth = 1
            return

        if self.in_body_section and tag == "section":
            self.body_section_depth += 1

        if not self.in_body_section and tag == "h1" and "content-title" in class_value:
            self.capture_title = True

        if self.in_body_section and tag == "p":
            self.capture_text = True
            self.current_text_parts = []

    def handle_endtag(self, tag: str):
        if self.capture_title and tag == "h1":
            self.capture_title = False

        if self.capture_text and tag == "p":
            paragraph = _normalize_xml_text(unescape("".join(self.current_text_parts)))
            if paragraph:
                self.paragraphs.append(paragraph)
            self.capture_text = False
            self.current_text_parts = []

        if self.in_body_section and tag == "section":
            self.body_section_depth -= 1
            if self.body_section_depth <= 0:
                self.in_body_section = False
                self.body_section_depth = 0

    def handle_data(self, data: str):
        if self.capture_title:
            self.title_parts.append(data)
        if self.capture_text:
            self.current_text_parts.append(data)


def _parse_pmc_html(html_content: str, pmid: str, pmcid: str) -> Dict[str, str] | None:
    """Parse PMC HTML when the XML payload contains only metadata."""
    parser = _PMCBodyHTMLParser()
    parser.feed(html_content)

    full_text = _normalize_xml_text(" ".join(parser.paragraphs))
    if not full_text:
        return None

    title = _normalize_xml_text("".join(parser.title_parts))
    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "title": title,
        "abstractText": full_text,
        "content_source": "pmc_fulltext_fallback",
    }


def fetch_pmc_fulltext_records(
    pmid_to_pmcid: Dict[str, str]
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Fetch PMC full-text fallbacks for PMIDs whose PubMed record lacks an abstract."""
    if not pmid_to_pmcid:
        return {}, {}

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    records: Dict[str, Dict] = {}
    failures: Dict[str, str] = {}

    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
        headers=DEFAULT_HTTP_HEADERS,
    ) as client:
        for pmid, pmcid in pmid_to_pmcid.items():
            if not pmcid:
                failures[pmid] = "no_pmcid"
                continue

            params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
            response = _request_with_retries(
                client=client,
                url=base_url,
                params=params,
                request_label=f"PMC full text for {pmcid}",
            )
            if response is None:
                failures[pmid] = "pmc_fetch_failed"
                continue

            record = _parse_pmc_xml(response.text, pmid=pmid, pmcid=pmcid)
            if record is None:
                html_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
                html_response = _request_with_retries(
                    client=client,
                    url=html_url,
                    params={},
                    request_label=f"PMC HTML full text for {pmcid}",
                )
                if html_response is None:
                    failures[pmid] = "pmc_html_fetch_failed"
                    continue

                record = _parse_pmc_html(html_response.text, pmid=pmid, pmcid=pmcid)
                if record is None:
                    failures[pmid] = "pmc_html_parse_failed"
                    continue

            records[pmid] = record
            time.sleep(0.4)

    return records, failures


def preprocess_bioasq_taskB(input_file: str, num_samples_needed: int = 1000):
    """
    Process training10b.json to generate a clean dataset for chunking and evaluation.
    This version ensures that every QA sample selected has all its referenced
    articles (with abstracts) available in the corpus.
    """
    dirs = setup_directories()
    
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_questions = json.load(f).get("questions", [])
    
    # --- Phase 1: Load existing corpus PMIDs ---
    corpus_path = dirs["corpus"] / "corpus.jsonl"
    existing_articles = {}
    if corpus_path.exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line)
                    if "pmid" in article and "abstractText" in article:
                        existing_articles[article["pmid"]] = article
                except json.JSONDecodeError:
                    continue
    print(f"Found {len(existing_articles)} existing articles in {corpus_path}")

    # --- Phase 2: Iteratively find 1000 fully valid samples ---
    valid_qa_samples = []
    articles_to_append = {} # Store new articles to be added to corpus
    
    print(f"Scanning for {num_samples_needed} valid samples with full abstract coverage...")
    
    for q in all_questions:
        if len(valid_qa_samples) >= num_samples_needed:
            print(f"\nSuccessfully collected {num_samples_needed} valid samples.")
            break

        # Basic structural validation
        if not (q.get("id") and q.get("body") and q.get("documents") and q.get("ideal_answer")):
            continue

        # Extract PMIDs and clean snippets
        pmids_in_question = {p for d in q.get("documents", []) if (p := extract_pmid(d))}
        
        clean_snippets = []
        for s in q.get("snippets", []):
            if s.get("text") and (s_pmid := extract_pmid(s.get("document"))):
                clean_snippets.append({"text": s["text"], "pmid": s_pmid})
                pmids_in_question.add(s_pmid)

        if not pmids_in_question:
            continue

        # Create a temporary sample to check for snippet coverage
        temp_sample = {
            "id": q["id"],
            "body": q["body"],
            "relevant_pmid": list(pmids_in_question),
            "snippets": clean_snippets,
            "ideal_answer": q["ideal_answer"]
        }

        if not _has_full_snippet_coverage(temp_sample):
            continue

        # --- Core Logic: Check for abstract availability for all PMIDs ---
        all_pmids_found_with_abstract = True
        pmids_to_fetch = []
        
        # Check which PMIDs we need to fetch
        for pmid in pmids_in_question:
            if pmid not in existing_articles and pmid not in articles_to_append:
                pmids_to_fetch.append(pmid)
        
        # Fetch any missing PMIDs
        if pmids_to_fetch:
            print(f"\nSample {q['id']} needs {len(pmids_to_fetch)} new PMIDs. Fetching...")
            fetched_data = fetch_pubmed_data(pmids_to_fetch)
            
            # Add successfully fetched articles to our temporary new articles dict
            for pmid, data in fetched_data.items():
                if data.get("abstractText"):
                    articles_to_append[pmid] = data
        
        # Final check: Do all PMIDs for this question now have an abstract?
        for pmid in pmids_in_question:
            # Check in existing corpus, then in newly fetched articles
            if pmid in existing_articles:
                if not existing_articles[pmid].get("abstractText"):
                    all_pmids_found_with_abstract = False
                    break
            elif pmid in articles_to_append:
                if not articles_to_append[pmid].get("abstractText"):
                    all_pmids_found_with_abstract = False
                    break
            else: # PMID was not found in existing corpus OR in the new fetch
                all_pmids_found_with_abstract = False
                break
        
        # If all good, add the sample to our final list
        if all_pmids_found_with_abstract:
            valid_qa_samples.append(temp_sample)
            print(f"Found {len(valid_qa_samples)}/{num_samples_needed} valid samples...", end="\r")

    # --- Phase 3: Write results to files ---
    if len(valid_qa_samples) < num_samples_needed:
        print(f"\nWarning: Could only find {len(valid_qa_samples)} samples that met all criteria.")

    # Append new articles to corpus.jsonl
    if articles_to_append:
        with open(corpus_path, 'a', encoding='utf-8') as f:
            for pmid_data in articles_to_append.values():
                f.write(json.dumps(pmid_data, ensure_ascii=False) + "\n")
        print(f"\nAppended {len(articles_to_append)} new articles to {corpus_path}")

    # Save the final QA samples
    qa_path = dirs["corpus"] / "bioasq_QA.jsonl"
    with open(qa_path, 'w', encoding='utf-8') as f:
        for sample in valid_qa_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(valid_qa_samples)} questions to {qa_path}")


def split_bioasq_taskB():
    """Split corpus_QA.jsonl into validation and test sets.

    Only keeps questions where every relevant PMID has at least one
    corresponding snippet, ensuring snippet-level evaluation is valid.
    Outputs are saved as val_bioasq.jsonl and test_bioasq.jsonl.
    """
    dirs = setup_directories()
    qa_path = dirs["corpus"] / "bioasq_QA.jsonl"

    if not qa_path.exists():
        print(f"Error: {qa_path} not found.")
        return

    samples = []
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    if len(samples) < 1000:
        print(f"Warning: Only found {len(samples)} samples. Expected 1000.")

    # Split directly since all samples are already filtered
    test_samples = samples[:500]
    val_samples = samples[500:1000]

    test_path = dirs["test"] / "test_bioasq.jsonl"
    with open(test_path, 'w', encoding='utf-8') as f:
        for s in test_samples:
            # ensure_ascii = False -> Ensure Vietnamese won't be encoded into Unicode escape
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    val_path = dirs["val"] / "val_bioasq.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Split complete. Saved {len(test_samples)} to {test_path} "
          f"and {len(val_samples)} to {val_path}")


if __name__ == "__main__":
    input_file = "data/raw/training10b.json"
    preprocess_bioasq_taskB(input_file)
    split_bioasq_taskB()
