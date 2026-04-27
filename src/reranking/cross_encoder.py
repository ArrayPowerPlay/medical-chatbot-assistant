from typing import List, Dict
import modal
import random
from config.logging_config import logger, setup_logging


class CrossEncoderReranker:
    """
    Client call to MedCPT-Cross-Encoder deployed on Modal.
    Merges text search results (keyword + vector search) and KG search results, then reranks them.
    """
    def __init__(self):
        try:
            # Link to deployed App on Modal
            self.model = modal.Cls.lookup("medcpt-cross-encoder-v1", "CrossEncoderModel")()   # type: ignore
            self.is_available = True
        except Exception as e:
            logger.error(f"[Cross Encoder Reranker]: Failed to lookup Modal App. Error: {e}")
            self.is_available = False

    def rerank(
        self,
        query: str,
        rrf_results: List[Dict],
        kg_text: str,
        top_k: int = 20
    ) -> List[Dict]:   # type: ignore
        """
        Reranks both text and KG passages.

        Args:
            query: The rewritten query of the user
            rrf_results: List of dicts from RRF
            kg_text: The linearized string from KG search (joined by newline)
            top_k: Number of final results to return
        
        Returns:
            A unified list of dictionaries sorted by 'cross_encoder_score'
        """
        if not self.is_available:
            logger.warning("[Cross Encoder Reranker]: Model unavailable. Falling back to random shuffle" \
            "of text and KG search.")
            fallback_results = []
            
            for item in rrf_results:
                doc = item.copy()
                doc["source_type"] = "text_retrieval"
                doc["cross_encoder_score"] = doc.get("rrf_score", 0.0)
                fallback_results.append(doc)

            if kg_text:
                lines = [line.strip() for line in kg_text.split("\n") if line.strip()]
                chunk_size = 10
                for i in range(0, len(lines), chunk_size):
                    kg_block = "\n".join(lines[i: i + chunk_size])
                    fallback_results.append({
                        "text": kg_block,
                        "cross_encoder_score": 0.0,        # Default set to 0.0
                        "source type": "kg_retrieval"
                    })

            # Random shuffle
            random.shuffle(fallback_results)
            return fallback_results[:top_k]
        
        passages = []
        mapping = []

        for item in rrf_results:
            passages.append(item["text"])
