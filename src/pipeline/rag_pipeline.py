"""
This module defines the main end-to-end RAG pipeline for the medical chatbot. It 
orchestrates the entire process from receiving the user query to generating the
final answer.

The pipeline follows these main steps:
    1. Query Analyzer: The initial query is rewritten, and medical entities and user
    intent are extracted.
    2. Parallel Retrieval: Three retrieval methods (Keyword/Vector/KG search) are 
    executed in parallel to gather relevant context.
    3. Fusion and reranking: Results from text-based retrieval (Keyword/Vector search)
    are first fused using RRF. Then, this fused list, along with the KG linearized results
    are reranked by MedCPT-Cross-Encoder.
    4. KG Merger: Results from KG are merged to avoid context redundant.
    4. Contextual Prompting: The top-ranked results are organized into a structured prompt
    for the generator model, using a head-tail placement strategy.
    5. Generation: An LLM generates the final, answer based on the provided context and
    the conversation history.
"""
from typing import Dict, Optional, List, Any
from operator import itemgetter     # Utility function used for getting items in Dict, List,...

from config.settings import settings
from config.logging_config import logger
from src.query.query_analyzer import QueryAnalyzer
from src.retrieval.parallel_retrieval import ParallelRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.rrf import RRFManager
from src.generation.kg_merger import KGPathMerger
from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import build_prompts
from src.storage.parent_store import ParentStore
from src.storage.weaviate_client import WeaviateChildStore
from kg.kg_search import KGSearch
from src.embeddings.medcpt_embedder import MedCPTEmbedder


class RAGPipeline:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.query_embedder = MedCPTEmbedder(mode='query')
        self.entity_embedder = MedCPTEmbedder(mode='article')

        self.weaviate_store = WeaviateChildStore()
        self.parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)
        self.kg_searcher = KGSearch()

        self.parallel_retriever = ParallelRetriever(
            weaviate_store=self.weaviate_store,
            parent_store=self.parent_store,
            kg_searcher=self.kg_searcher
        )

        self.rrf_manager = RRFManager()
        self.cross_encoder_reranker = CrossEncoderReranker()
        self.kg_merger = KGPathMerger()
        self.llm_generator = LLMGenerator()

    def _normalize_history(
        self,
        history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Normalize conversation history into the message schema used by Groq.
        Return a cleaned list of role/content messages limited to the lastest
        turns (5 turns) that the model should use."""
        if not history:
            return []
        
        normalized: List[Dict[str, str]] = []
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role not in ("system", "user", "assistant"):
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            normalized.append({"role": role, "content": content.strip()})
        return normalized[-5:]
    
    def _build_entity_texts(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Build entity texts for MedCPT-Article-Encoder embedding.

        Args:
            analysis: Result dictionary from QueryAnalyzer.

        Returns:
            A list of entity string enriched with their coarse type labels.
        """
        entity_texts: List[str] = []
        
        for disease in analysis.get("diseases", []):
            if isinstance(disease, str) and disease.strip():
                entity_texts.append(f"Disease: {disease.strip()}")
        
        for drug in analysis.get("drugs", []):
            if isinstance(drug, str) and drug.strip():
                entity_texts.append(f"Drug: {drug.strip()}")

        for ep in analysis.get("effect_phenotypes", []):
            if isinstance(ep, str) and ep.strip():
                entity_texts.append(f"EffectPhenotype: {ep.strip()}")

        for gp in analysis.get("gene_proteins", []):
            if isinstance(gp, str) and gp.strip():
                entity_texts.append(f"GeneProtein": gp.strip())

        return entity_texts
    
    def _build_canonical_terms(self, ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert reranked output into the canonical prompt schema."""