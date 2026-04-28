from typing import List, Dict
import modal
from config.settings import settings
from config.logging_config import logger


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
        top_m: int = settings.RERANK_TEXT_TOP_M,
        top_n: int = settings.RERANK_KG_TOP_N
    ) -> List[Dict]:   # type: ignore
        """
        Reranks both text and KG passages.

        Args:
            query: The rewritten query of the user
            rrf_results: List of dicts from RRF
            kg_text: The linearized string from KG search (joined by newline)
            top_m: Number of text results to return
            top_n: Number of KG results to return
        
        Returns:
            A unified list of dictionaries sorted by 'cross_encoder_score'
        """
        if not self.is_available:
            logger.warning("Cannot connect to Modal App, return random ranking results.")
            return rrf_results[:(top_m + top_n)]
        
        passages = []
        mapping = []

        # Append text passages
        for item in rrf_results:
            passages.append(item["text"])
            mapping.append({"type": "text", "data": item})

        # Append KG passages
        if kg_text:
            lines = [line.strip() for line in kg_text.split("\n") if line.strip()]
            for line in lines:
                passages.append(line)
                mapping.append({"type": "kg", "data": line})

        if not passages:
            return []
        
        try:
            scores = self.model.rerank.remote(query, passages)
        except Exception as e:
            logger.error(f"[Cross Encoder]: Rerank failed: {e}")
            return rrf_results[:(top_m + top_n)]
        
        # Filter scores >= 0
        scored_text = []
        scored_kg = []

        for i, score in enumerate(scores):
            if score < 0: continue
            
            mapped_item = mapping[i]
            if mapped_item["type"] == "text":
                doc = mapped_item["data"].copy()
                doc["cross_encoder_score"] = score
                doc["source_type"] = "text_retrieval"
                scored_text.append(doc)
            else:
                doc = {
                    "text": mapped_item["data"],
                    "cross_encoder_score": score,
                    "source_type": "kg_retrieval"
                }
                scored_kg.append(doc)

        # Sort each modality and apply top_m/top_n filters
        scored_text.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        scored_kg.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        final_results = scored_text[:top_m] + scored_kg[:top_n]

        # Sort unified list
        final_results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return final_results
        
