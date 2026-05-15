from typing import List, Dict
from config.settings import settings


class RRFManager:
    """
    Manages Reciprocal Rank Fusion (RRF) to merge multiple retrieval streams
    """
    def __init__(self, k: int = settings.K_RRF):
        if k < 0:
            raise ValueError(f"RRF parameter k must be >= 0, got {k}")
        self.k = k

    def rank_fusion(
        self,
        vector_results: List[Dict],  
        bm25_results: List[Dict],
        top_k: int = settings.TOP_K_RRF
    ) -> List[Dict]:
        """
        Merge Vector and BM25 search results using Reciprocal Rank Fusion
        Args:
            vector_results: List of results from semantic search, sorted by score
            bm25_results: List of results from keyword search, sorted by score
            top_k: Number of fused results to return
        Returns:
            A fused and re-ranked list of parent chunks
        """
        scores: Dict[str, float] = {}
        # Metadata storage to preserve parent chunk details (pmid, text, etc.)
        metadata_storage: Dict[str, Dict] = {}

        # Process 'score' from vector database
        for rank, item in enumerate(vector_results, 1):
            parent_id = item["parent_id"]
            if parent_id not in scores:
                scores[parent_id] = 0.0
                metadata_storage[parent_id] = item

            scores[parent_id] += 1.0 / (self.k + rank)
        
        # Process 'score' from BM25 database
        for rank, item in enumerate(bm25_results, 1):
            parent_id = item["parent_id"]
            if parent_id not in scores:
                scores[parent_id] = 0.0
                metadata_storage[parent_id] = item

            scores[parent_id] += 1.0 / (self.k + rank)

        # Convert scores to a sorted list
        fused_results = []
        for parent_id, rrf_score in scores.items():
            item = metadata_storage[parent_id].copy()
            item['rrf_score'] = rrf_score
            item['source_type'] = 'text_retrieval'
            if 'score' in item:
                del item['score']
            fused_results.append(item)
        
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        return fused_results[:top_k]
