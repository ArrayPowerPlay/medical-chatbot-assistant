import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict
import httpx

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


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


def fetch_pubmed_data(pmids: List[str]) -> Dict[str, Dict]:
    """Fetch abstract and title from PubMed using NCBI E-Utilities"""
    if not pmids: return {}
    
    # NCBI limit: 3 requests per second without API key. Batching is key.
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    results = {}
    
    # Process in batches of 200 to avoid long URLs
    batch_size = 150
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
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(base_url, params=params)
                if response.status_code == 200:
                    content = response.text
                    
                    # Simple XML parsing using regex for speed/minimal dependencies
                    # In a production environment, use lxml or xml.etree
                    articles = re.findall(r'<PubmedArticle>.*?</PubmedArticle>', content, re.DOTALL)
                    for article in articles:
                        pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', article)
                        title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', article, re.DOTALL)
                        abstract_match = re.search(r'<AbstractText>(.*?)</AbstractText>', article, re.DOTALL)
                        
                        if pmid_match:
                            p = pmid_match.group(1)
                            title = title_match.group(1).strip() if title_match else ""
                            # Abstract can have multiple parts, joining them
                            abstract = ""
                            if abstract_match:
                                # Remove internal XML tags from title/abstract if any
                                title = re.sub(r'<[^>]*>', '', title)
                                abstract_parts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', article, re.DOTALL)
                                abstract = " ".join([re.sub(r'<[^>]*>', '', part).strip() for part in abstract_parts])
                            
                            results[p] = {
                                "pmid": p,
                                "title": title,
                                "abstractText": abstract
                            }

                            if len(results) % 5 == 0:
                                    print(f"Fetched {len(results)}/{total_pmids} articles...", end="\r")

                time.sleep(0.4) # Respect NCBI rate limits
        except Exception as e:
            print(f"Error fetching PubMed data: {e}")
            
    return results


def preprocess_bioasq_taskB(input_file: str):
    """Process training10b.json and generate dataset used for chunking and evaluation"""
    dirs = setup_directories()
    
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    valid_qa_samples = []
    
    # We need to collect all PMIDs first to fetch abstracts efficiently
    all_pmids_needed = set()
    
    print("Scanning for valid samples and PMIDs...")
    raw_index = 0
    
    # Phase 1: Identify 1000 valid samples
    while len(valid_qa_samples) < 1000 and raw_index < len(questions):
        q = questions[raw_index]
        raw_index += 1
        
        try:
            # Check if sample is valid (has required fields)
            q_id = q.get("id")
            body = q.get("body")
            docs = q.get("documents", [])
            snippets = q.get("snippets", [])
            ideal_answer = q.get("ideal_answer")
            
            if not (q_id and body and docs and ideal_answer):
                continue
                
            pmids = [extract_pmid(d) for d in docs if extract_pmid(d)]
            if not pmids:
                continue
                
            # Process snippets
            clean_snippets = []
            for s in snippets:
                s_text = s.get("text")
                s_doc_url = s.get("document")
                s_pmid = extract_pmid(s_doc_url)
                if s_text and s_pmid:
                    clean_snippets.append({"text": s_text, "pmid": s_pmid})
            
            sample = {
                "id": q_id,
                "body": body,
                "relevant_pmid": pmids,
                "snippets": clean_snippets,
                "ideal_answer": ideal_answer
            }
            
            valid_qa_samples.append(sample)
            all_pmids_needed.update(pmids)
            for cs in clean_snippets:
                all_pmids_needed.add(cs["pmid"])
                
        except Exception:
            continue # Skip invalid samples
            
    print(f"Collected {len(valid_qa_samples)} valid questions. Fetching abstracts for {len(all_pmids_needed)} unique PMIDs...")
    
    # Phase 2: Fetch PubMed data for all unique PMIDs
    pubmed_data = fetch_pubmed_data(list(all_pmids_needed))
    
    # Phase 3: Save corpus.jsonl - used for chunking
    corpus_path = dirs["corpus"] / "corpus.jsonl"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for pmid_data in pubmed_data.values():
            f.write(json.dumps(pmid_data, ensure_ascii=False) + "\n")
            
    # Phase 4: Save corpus_QA.jsonl - used for evaluation
    qa_path = dirs["corpus"] / "corpus_QA.jsonl"
    with open(qa_path, 'w', encoding='utf-8') as f:
        for sample in valid_qa_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    print(f"Preprocessing complete. Saved {len(pubmed_data)} articles to {corpus_path} and 1000 questions to {qa_path}")


def split_bioasq_taskB():
    """Split corpus_QA.jsonl into validation set (used for tuning hyperparameters) and test set"""
    dirs = setup_directories()
    qa_path = dirs["corpus"] / "corpus_QA.jsonl"
    
    if not qa_path.exists():
        print(f"Error: {qa_path} not found.")
        return
        
    samples = []
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
            
    if len(samples) < 1000:
        print(f"Warning: Only found {len(samples)} samples. Expected 1000.")
        
    test_samples = samples[:500]
    val_samples = samples[500:1000]
    
    test_path = dirs["test"] / "test.jsonl"
    with open(test_path, 'w', encoding='utf-8') as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    val_path = dirs["val"] / "val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"Split complete. Saved 500 samples to {test_path} and {len(val_samples)} samples to {val_path}")


if __name__ == "__main__":
    input_file = "data/raw/training10b.json"
    preprocess_bioasq_taskB(input_file)
    split_bioasq_taskB()
