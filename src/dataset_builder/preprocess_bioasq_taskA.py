import json
import os
import sys
import time
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

# Configure project root to import from config module
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings

# Prefix tree number — use after map "meshMajor" to id in "data/mtrees2024.bin"
DISEASE_PREFIXES = [
    "C01", "C02", "C03", "C04", "C05", "C06", "C07",
    "C08", "C09", "C10", "C11", "C12", "C13", "C14",
    "C15", "C16", "C17", "C18", "C19", "C20",
    "C21",  
    "C23", 
    "C25",  
]

DRUG_TARGET_PREFIXES = [
    "D03", "D04",   
    "D06",        
    "D08",         
    "D12",        
    "D13",         
    "D23",      
    "D26",       
    "D27",          
]

ALL_PREFIXES = DISEASE_PREFIXES + DRUG_TARGET_PREFIXES


def load_mesh_mapping(mapping_file: str):
    """Load the MeSH Name to Tree Numbers mapping from mtrees2024.bin"""
    mapping = {}
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file {mapping_file} not found.")
        return mapping
        
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ';' not in line:
                continue
            try:
                name, tree_num = line.split(';', 1)
                if name not in mapping:
                    mapping[name] = []
                mapping[name].append(tree_num)
            except ValueError:
                continue
    return mapping


def get_existing_pmids(corpus_path: Path):
    """Get a set of PMIDs already in the corpus file to avoid duplicates"""
    pmids = set()
    if corpus_path.exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'pmid' in data:
                        pmids.add(str(data['pmid']))
                except json.JSONDecodeError:
                    continue
    return pmids


def download_pubmed_corpus():
    """Download 300,000 articles from 'jmhb/pubmed_bioasq_2022' (dataset from huggingface) with filtering"""
    corpus_dir = Path("data/corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "corpus.jsonl"
    mapping_file = "data/mtrees2024.bin"
    
    # 0. Login into Hugging Face
    if os.getenv("HF_TOKEN"):
        pass
    elif hasattr(settings, "HF_TOKEN") and settings.HF_TOKEN:
        print("Logging into Hugging Face...")
        try:
            login(token=settings.HF_TOKEN)
        except Exception as e:
            print(f"Warning: Hugging Face login failed: {e}. Proceeding as guest...")
    else:
        print("HF_TOKEN not found or empty. Proceeding as guest...")
    
    # 1. Load MeSH mapping
    print("Loading MeSH mapping from mtrees2024.bin...")
    mesh_map = load_mesh_mapping(mapping_file)
    
    # 2. Get existing PMIDs for deduplication
    existing_pmids = get_existing_pmids(corpus_path)
    
    # 3. Stream dataset from Hugging Face
    print("Streaming dataset jmhb/pubmed_bioasq_2022 from Hugging Face...")
    try:
        dataset = load_dataset(
            "jmhb/pubmed_bioasq_2022", 
            data_files="data/allMeSH_2022.parquet",
            split="train", 
            streaming=True,
            token=settings.HF_TOKEN
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    target_count = 300000
    downloaded_count = 0
    
    # 4. Open corpus.jsonl in append mode
    print(f"Processing data in manual batches of 100 with time delay...")
    with open(corpus_path, 'a', encoding='utf-8') as f:
        try:
            dataset_iter = iter(dataset)
            while True:
                # Manual batching: collect up to 100 items
                batch = []
                try:
                    for _ in range(100):
                        batch.append(next(dataset_iter))
                except StopIteration:
                    if not batch:
                        break # End of dataset
                
                # Process each article in the batch
                for art in batch:
                    pmid = str(art.get("pmid", ""))
                    
                    # Check for duplicates using the full set of PMIDs
                    if not pmid or pmid in existing_pmids:
                        continue
                    
                    # Check MeSH filter
                    mesh_major = art.get("meshMajor", [])
                    is_relevant = False
                    for mesh_name in mesh_major:
                        tree_numbers = mesh_map.get(mesh_name, [])
                        for tn in tree_numbers:
                            if any(tn.startswith(prefix) for prefix in ALL_PREFIXES):
                                is_relevant = True
                                break
                        if is_relevant:
                            break
                    
                    if is_relevant:
                        # Construct record
                        record = {
                            "abstractText": art.get("abstractText", ""),
                            "pmid": pmid,
                            "title": art.get("title", "")
                        }
                        
                        # Write to file
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        downloaded_count += 1
                        existing_pmids.add(pmid)
                        
                        # Log progress every 100 successful downloads
                        if downloaded_count % 100 == 0:
                            print(f"Downloaded {downloaded_count}/{target_count} samples...", end="\r")
                    
                    if downloaded_count >= target_count:
                        break
                
                # Add delay after each batch
                time.sleep(0.5)
                
                if downloaded_count >= target_count:
                    break
        except Exception as e:
            print(f"\nStopped due to error during streaming: {e}")
                
    print(f"\nFinished! Added {downloaded_count} new samples to {corpus_path}")


if __name__ == "__main__":
    download_pubmed_corpus()
