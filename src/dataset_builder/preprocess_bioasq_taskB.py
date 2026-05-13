import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict
import httpx
import xml.etree.ElementTree as ET

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


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


def _parse_pubmed_xml(xml_content: str) -> Dict[str, Dict]:
    """Parse the XML response from PubMed and extract article details."""
    results = {}
    try:
        root = ET.fromstring(xml_content)
        for article in root.findall('.//PubmedArticle'):
            pmid_element = article.find('.//PMID')
            if pmid_element is None or pmid_element.text is None:
                continue
            
            pmid = pmid_element.text
            
            title_element = article.find('.//ArticleTitle')
            # Ensure title is a string, default to empty string if None
            title = title_element.text if title_element is not None and title_element.text is not None else ""
            
            abstract_texts = []
            for abstract_element in article.findall('.//AbstractText'):
                if abstract_element.text:
                    abstract_texts.append(abstract_element.text)
            abstract = " ".join(abstract_texts)
            
            # Clean up potential HTML tags just in case
            title = re.sub(r'<[^>]*>', '', title)
            abstract = re.sub(r'<[^>]*>', '', abstract)

            results[pmid] = {
                "pmid": pmid,
                "title": title.strip(),
                "abstractText": abstract.strip()
            }
    except ET.ParseError as e:
        print(f"\nError parsing XML: {e}")
    return results


def fetch_pubmed_data(pmids: List[str]) -> Dict[str, Dict]:
    """Fetch abstract and title from PubMed using NCBI E-Utilities with retries."""
    if not pmids:
        return {}
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    results = {}
    failed_pmids = []
    
    batch_size = 150
    max_retries = 3
    total_pmids = len(pmids)
    print(f"Starting to fetch abstracts for {total_pmids} unique PMIDs...")

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(base_url, params=params)
                    response.raise_for_status()  # Will raise an exception for 4xx/5xx status
                    
                    # Successfully fetched, now parse
                    batch_results = _parse_pubmed_xml(response.text)
                    results.update(batch_results)
                    
                    # Check if all PMIDs in the batch were parsed successfully
                    missing_in_batch = set(batch) - set(batch_results.keys())
                    if missing_in_batch:
                        print(f"\nWarning: Could not parse data for PMIDs in batch: {list(missing_in_batch)}")
                        # You might want to add them to a retry list, but often this is a parsing/data issue
                    
                    print(f"Fetched {len(results)}/{total_pmids} articles...", end="\r")
                    time.sleep(0.4) # Respect NCBI rate limits
                    break # Exit retry loop on success

            except httpx.RequestError as e:
                print(f"\nError fetching batch starting at index {i}. Attempt {attempt + 1}/{max_retries}. Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    print(f"\nFailed to fetch batch after {max_retries} attempts. Skipping PMIDs: {batch}")
                    failed_pmids.extend(batch)
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                # Decide if you want to retry on unexpected errors
                break # Or continue to next batch

    if failed_pmids:
        print("\n--------------------------------------------------")
        print(f"Could not fetch data for the following {len(failed_pmids)} PMIDs after all retries:")
        print(", ".join(failed_pmids))
        print("--------------------------------------------------")
            
    return results


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
