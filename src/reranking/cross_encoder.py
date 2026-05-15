from typing import List, Dict, Tuple
import modal
from config.settings import settings
from config.logging_config import logger


class CrossEncoderReranker:
    """
    Client call to MedCPT-Cross-Encoder deployed on Modal.
    Merges text search results (keyword + vector search) and KG search results, then reranks them.
    """
    def __init__(self):
        self.model = None
        try:
            # Link to deployed App on Modal
            self.model = modal.Cls.from_name("medcpt-cross-encoder-v1", "CrossEncoderModel")()   # type: ignore
            self.is_available = True
        except Exception as e:
            logger.error(f"[Cross Encoder Reranker]: Failed to lookup Modal App. Error: {e}")
            self.is_available = False

    def rerank(
        self,
        query: str,
        rrf_results: List[Dict],
        kg_results: List[Dict],
        top_m: int = settings.RERANK_TEXT_TOP_M,
        top_n: int = settings.RERANK_KG_TOP_N
    ) -> Tuple[List[Dict], List[Dict]]:   # type: ignore
        """
        Reranks both text and KG passages.

        Args:
            query: The rewritten query of the user
            rrf_results: List of dicts from RRF
            kg_results: List of dictionary paths from KG search with metadata
            top_m: Number of text results to return
            top_n: Number of KG results to return
        
        Returns:
            A tuple of (ranked_text, ranked_kg), where each list is sorted independently.
        """
        def _normalize_text_doc(doc: Dict, score: float) -> Dict:
            normalized = doc.copy()
            normalized['cross_encoder_score'] = score
            normalized['source_type'] = 'text_retrieval'
            return normalized

        def _normalize_kg_doc(kg_doc: Dict, score: float) -> Dict:
            return {
                'text': kg_doc['text'],
                'cross_encoder_score': score,
                'source_type': 'kg_retrieval',
                'metadata': kg_doc.get('metadata', {}),
            }

        def _fallback_rankings() -> Tuple[List[Dict], List[Dict]]:
            fallback_text = []
            for item in rrf_results[:top_m]:
                fallback_item = item.copy()
                fallback_item["source_type"] = "text_retrieval"
                fallback_text.append(fallback_item)

            fallback_kg = []
            for item in kg_results[:top_n]:
                fallback_kg.append(_normalize_kg_doc(item, 0.0))

            return fallback_text, fallback_kg

        if not self.is_available:
            logger.warning("Cannot connect to Modal App, returning fallback ranking results.")
            return _fallback_rankings()
        
        passages = []
        mapping = []

        # Append text passages
        for item in rrf_results:
            passages.append(item["text"])
            mapping.append({"type": "text", "data": item})

        # Append KG passages
        if kg_results:
            for item in kg_results:
                passages.append(item["text"])
                mapping.append({"type": "kg", "data": item})

        if not passages:
            return [], []
        
        try:
            scores = self.model.rerank.remote(query, passages)       # type: ignore
        except Exception as e:
            logger.error(f"[Cross Encoder]: Rerank failed: {e}")
            return _fallback_rankings()
        
        # Filter scores > 0
        scored_text = []
        scored_kg = []

        for i, score in enumerate(scores):
            mapped_item = mapping[i]
            if mapped_item["type"] == "text":
                scored_text.append(_normalize_text_doc(mapped_item["data"], score))
            else:
                scored_kg.append(_normalize_kg_doc(mapped_item["data"], score))

        # Sort each modality and apply top_m/top_n filters
        scored_text.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        scored_kg.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        
        return scored_text[:top_m], scored_kg[:top_n]

    def close(self) -> None:
        """Best-effort cleanup for Modal sync wrappers and their event loop."""
        model = getattr(self, "model", None)
        if model is None:
            return

        synchronizer = getattr(model, "_sync_synchronizer", None)
        if synchronizer is not None and hasattr(synchronizer, "_close_loop"):
            try:
                synchronizer._close_loop()
            except Exception as e:
                logger.debug(f"[Cross Encoder]: Failed to close Modal synchronizer cleanly: {e}")

        self.model = None
        self.is_available = False
        
