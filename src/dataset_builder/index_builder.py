import faiss
import numpy as np
from elasticsearch import Elasticsearch, helpers
from typing import List, Dict
import json
from pathlib import Path


class IndexBuilder:
    """Built for pushing vector embeddings into FAISS and texts into ElasticSearch"""
    def __init__(self, es_url: str = "http://localhost:9200"):
        self.es = Elasticsearch(es_url)
        self.faiss_index: faiss.Index | None = None

    def build_faiss(self, embeddings: np.ndarray):
        """Build FAISS index (support semantic search)"""
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)         # Use cosine similarity
        faiss.normalize_L2(embeddings)
        assert self.faiss_index is not None
        self.faiss_index.add(embeddings)                  # type: ignore

    def build_elasticsearch(self, chunks: List[Dict], index_name: str = "med_corpus"):
        """Build elastic search index with explicit mapping (keyword search)"""
        if not self.es.indices.exists(index=index_name):  #  Check if index exists
            # Define data type and indexing method
            mapping = {
                "mappings": {
                    "properties": {
                        "pmid": {"type": "keyword"},      # Used for exact match
                        "text": {
                            "type": "text",               # Tokenize, normalize (lowercase, stemming,...)
                            "analyzer": "english" 
                        },
                    }
                }
            }
            self.es.indices.create(index=index_name, body=mapping)

        # Define a list of documents that would be inserted into Elastic Search
        actions = [
            {
                "_index": index_name,
                "_id": f"{chunk.get('pmid', 'unknown')}_{i}",
                "_source": chunk
            }
            for i, chunk in enumerate(chunks)
        ]
        
        helpers.bulk(self.es, actions)    # Support bulk insert
    
    def save_index(self, path: str):
        """Save FAISS index into file"""
        assert self.faiss_index is not None
        faiss.write_index(self.faiss_index, path)
