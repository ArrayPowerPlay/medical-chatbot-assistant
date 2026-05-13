import httpx
import xml.etree.ElementTree as ET
import re
import json
import time
from typing import List, Dict

# List of PMIDs that failed to parse previously
FAILED_PMIDS = [
    "21250223", "20301628", "21249951", "23986914", "20301293", 
    "20301420", "23104528", "20301331", "20301416", "20301462", 
    "20301779", "24212220", "20301588", "23890950", "23658991", 
    "23833797", "20301466"
]

OUTPUT_FILE = "failed_pmids_corpus.jsonl"

def parse_special_pubmed_xml(xml_content: str) -> Dict[str, Dict]:
    """
    A more robust XML parser to handle special cases like GeneReviews,
    which have structured abstracts.
    """
    results = {}
    try:
        root = ET.fromstring(xml_content)
        for article_set in root.findall('.//PubmedArticleSet'): # Handle both book and article structures
             for article in article_set.findall('.//PubmedArticle') + article_set.findall('.//PubmedBookArticle'):
                pmid_element = article.find('.//PMID')
                if pmid_element is None or pmid_element.text is None:
                    continue
                
                pmid = pmid_element.text
                
                title_element = article.find('.//ArticleTitle')
                title = title_element.text if title_element is not None and title_element.text is not None else ""
                
                # This part is more robust for structured abstracts (e.g., with labels)
                # It finds all AbstractText nodes and joins their text content.
                abstract_parts = []
                for abstract_node in article.findall('.//Abstract/AbstractText'):
                    # .itertext() gets all inner text, including from child tags
                    text_content = ''.join(abstract_node.itertext())
                    if text_content:
                        # Add label if it exists, which is common in GeneReviews
                        label = abstract_node.get('Label', '')
                        if label:
                            abstract_parts.append(f"{label.title()}: {text_content.strip()}")
                        else:
                            abstract_parts.append(text_content.strip())

                abstract = "\n".join(abstract_parts)
                
                # Clean up any remaining XML/HTML tags and extra whitespace
                title = re.sub(r'<[^>]*>', '', title).strip()
                abstract = re.sub(r'<[^>]*>', '', abstract).strip()

                if pmid in FAILED_PMIDS:
                    print(f"Successfully parsed PMID: {pmid}")
                    results[pmid] = {
                        "pmid": pmid,
                        "title": title,
                        "abstractText": abstract
                    }
    except ET.ParseError as e:
        print(f"Error parsing XML content: {e}")
    return results

def fetch_and_parse_manually(pmids: List[str]):
    """Fetches and manually parses a list of PMIDs."""
    if not pmids:
        print("No PMIDs to process.")
        return

    print(f"Starting manual fetch for {len(pmids)} PMIDs...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract"
    }

    all_parsed_data = {}
    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()
            
            # The response contains XML for all requested PMIDs
            parsed_data = parse_special_pubmed_xml(response.text)
            all_parsed_data.update(parsed_data)

    except httpx.RequestError as e:
        print(f"Failed to fetch data. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Check which PMIDs were not found/parsed
    successfully_parsed_pmids = set(all_parsed_data.keys())
    still_failed = set(pmids) - successfully_parsed_pmids
    if still_failed:
        print("\n--------------------------------------------------")
        print(f"Could not parse the following PMIDs: {list(still_failed)}")
        print("They may not have an abstract or have a very unusual format.")
        print("--------------------------------------------------")

    # Save the successfully parsed data to a new file
    if all_parsed_data:
        print(f"\nWriting {len(all_parsed_data)} successfully parsed articles to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for pmid_data in all_parsed_data.values():
                f.write(json.dumps(pmid_data, ensure_ascii=False) + "\n")
        print("Done.")
        print(f"\nNext step: You can now append the contents of '{OUTPUT_FILE}' to your main 'data/corpus/corpus.jsonl' file.")
    else:
        print("No data was parsed successfully.")


if __name__ == "__main__":
    fetch_and_parse_manually(FAILED_PMIDS)
