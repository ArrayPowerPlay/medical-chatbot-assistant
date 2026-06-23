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
import argparse

from config.settings import settings
from config.logging_config import logger, setup_logging
from src.query.query_analyzer import QueryAnalyzer
from src.retrieval.parallel_retrieval import ParallelRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.rrf import RRFManager
from src.generation.kg_merger import KGPathMerger
from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import build_prompts
from src.storage.parent_store import ParentStore
from src.storage.weaviate_client import WeaviateChildStore
from src.kg.kg_search import KGSearch
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
        history: Optional[List[Dict[str, str]]],
        max_messages: Optional[int] = None
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

        if max_messages is not None: 
            return normalized[-max_messages:]
        return normalized
    
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
                entity_texts.append(f"GeneProtein: {gp.strip()}")

        return entity_texts
    
    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        conversation_id: Optional[str] = None,
        history_turns: int = settings.HISTORY_TURNS_FOR_LLM,
        vector_top_k: int = settings.VECTOR_TOP_K,
        keyword_top_k: int = settings.KEYWORD_TOP_K,
        child_fetch_limit: int = settings.CHILD_FETCH_LIMIT,
        kg_top_k: int = settings.KG_TOP_K,
        kg_hop1_m: int = settings.KG_HOP1_M,
        kg_hop2_n: int = settings.KG_HOP2_N,
        kg_hop2_cap: int = settings.KG_HOP2_CAP,
        rerank_kg_top_n: int = settings.RERANK_KG_TOP_N,
        generation_temperature: float = settings.GENERATION_TEMPERATURE,
        generation_max_tokens: int = settings.GENERATION_MAX_TOKENS,
        use_kg_merger: bool = settings.USE_KG_MERGER,
        use_head_tail_placement: bool = settings.USE_HEAD_TAIL_PLACEMENT,
        use_citations: bool = settings.USE_CITATIONS,
        question_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full synchronous RAG pipeline.
        
        Args:
            query: Raw user question.
            history: Optional conversation history for query rewriting and answer generation.
            conversation_id: Optional conversation identifier to carry through the response.
            top_k: Retrieval depth for vector and BM25 stages.
            child_fetch_limit: Number of child chunks fetched per text-retrieval stream.

        Returns:
            A dictionary with the answer, sources, rewritten query, entities, intents, and
            conversation_id.
        """
        max_messages = history_turns * 2
        normalized_history = self._normalize_history(history, max_messages)
        logger.info("[RAG Pipeline]: Starting query analysis...")
        
        ### 1. Extract entities and intents from user's question
        analysis = self.query_analyzer.analyze(query=query, history=normalized_history)
        rewritten_query = analysis.get("rewritten_query", query)
        intents = analysis.get("intents", ["general"])
        # question_type is classified by the LLM inside QueryAnalyzer (no extra call)
        question_type = analysis.get("question_type", "summary")

        query_vector = self.query_embedder.embed_texts(rewritten_query)[0]
        entity_texts = self._build_entity_texts(analysis)
        entity_artical_embeddings: List[List[float]] = []
        if entity_texts:
            entity_artical_embeddings = self.entity_embedder.embed_texts(entity_texts).tolist()

        ### 2. Start parallel retrieval
        logger.info("[RAG Pipeline]: Starting parallel retrieval...")
        vector_results, bm25_results, kg_results = self.parallel_retriever.retrieve(
            query_text=rewritten_query,
            query_vector=query_vector,
            entity_article_embeddings=entity_artical_embeddings,
            intents=intents,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
            child_fetch_limit=child_fetch_limit,
            kg_top_k=kg_top_k,
            kg_hop1_m=kg_hop1_m,
            kg_hop2_n=kg_hop2_n,
            kg_hop2_cap=kg_hop2_cap,
            original_query_text=query,
        )

        logger.info(f"[RAG Pipeline]: Parallel retrieval completed! "
                    f"{len(vector_results)} vectors, {len(bm25_results)} bm25, "
                    f"{len(kg_results) if kg_results else 0} KG paths.")

        ### 3. RRF on text sources
        logger.info(f"[RAG Pipeline] Running RRF on text retrieval stream...")
        rrf_results = self.rrf_manager.rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results
        )
        logger.info(f"[RAG Pipeline]: RRF produced {len(rrf_results)} fused candidates.")

        ### 4. Reranking on RRF results and KG results (linearized triples)
        logger.info(f"[RAG Pipeline]: Cross-Encoder reranking...")
        ranked_text, ranked_kg = self.cross_encoder_reranker.rerank(
            query=rewritten_query,
            rrf_results=rrf_results,
            kg_results=kg_results,
            top_n=rerank_kg_top_n,
        )
        logger.info(f"[RAG Pipeline]: Cross-Encoder returned {len(ranked_text)} texts and {len(ranked_kg)} KG paths.")
        logger.info(f"[RAG Pipeline]: Merging KG paths and preparing prompt context...")
        merged_kg = (
            self.kg_merger.merge_top_paths(ranked_kg)
            if use_kg_merger
            else ranked_kg
        )

        ### 5. Context re-ordering on text results and KG merged results
        # Manual interleaving
        interleaved_items = []
        max_len = max(len(merged_kg), len(ranked_text))
        
        for i in range(max_len):
            if i < len(ranked_text):
                interleaved_items.append(ranked_text[i])
            if i < len(merged_kg):
                interleaved_items.append(merged_kg[i])

        if not interleaved_items:
            logger.warning("[RAG Pipeline]: Pipeline returned empty context!")
        
        system_prompt, user_prompt = build_prompts(
            query=rewritten_query,
            retrieved_items=interleaved_items,
            use_head_tail_placement=use_head_tail_placement,
            use_citations=use_citations,
            question_type=question_type,
        )

        ### 6. Generate the final answer
        logger.info("[RAG Pipeline]: Generating final answer...")
        self.llm_generator.temperature = generation_temperature
        self.llm_generator.max_tokens = generation_max_tokens
        answer = self.llm_generator.generate_answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=normalized_history
        )

        return {
            "answer": answer,
            "conversation_id": conversation_id,
            "sources": interleaved_items,
            "rewritten_query": rewritten_query,
            "analysis": analysis,
            "question_type": question_type,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full RAG pipeline.")
    parser.add_argument("--question", type=str, help="Question you want to ask.")
    return parser


if __name__ == "__main__":
    setup_logging()
    pipeline = RAGPipeline()
    args = build_parser().parse_args()
    result = pipeline.run(query = args.question, history=None)
    print(result["rewritten_query"])