"""
End-to-end synchronous RAG pipeline orchestration for the medical chatbot.

This module wires together the existing sync components in the repository:
- Query rewriting and medical intent/entity extraction
- Dual-encoder embeddings for query and entities
- Parallel retrieval across vector, keyword, and KG streams
- RRF fusion for text retrieval only
- Cross-encoder reranking over unified text + KG candidates
- KG path merging and prompt construction
- Final answer generation with optional conversation history

The FastAPI layer can wrap this pipeline in a threadpool later, so the core
pipeline remains synchronous and simple to integrate with the current clients.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.logging_config import logger
from config.settings import settings
from src.embeddings.medcpt_embedder import MedCPTEmbedder
from src.generation.kg_merger import KGPathMerger
from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import build_prompts
from src.kg.kg_search import KGSearch
from src.query.query_analyzer import QueryAnalyzer
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.rrf import RRFManager
from src.retrieval.parallel_retrieval import ParallelRetriever
from src.storage.parent_store import ParentStore
from src.storage.weaviate_client import WeaviateChildStore


class RAGPipeline:
	"""Synchronous orchestration layer for the medical KG-RAG pipeline.

	The pipeline keeps all current clients in sync mode and performs parallel
	retrieval with a thread pool inside `ParallelRetriever`. This keeps the
	design aligned with the rest of the repository while still allowing the
	API layer to wrap execution in a threadpool later.
	"""

	def __init__(self) -> None:
		"""Initialize shared clients and processing components.

		Args:
			None

		Returns:
			None
		"""
		self.query_analyzer = QueryAnalyzer()
		self.query_embedder = MedCPTEmbedder(mode="query")
		self.entity_embedder = MedCPTEmbedder(mode="article")

		self.weaviate_store = WeaviateChildStore()
		self.parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)
		self.kg_searcher = KGSearch()

		self.parallel_retriever = ParallelRetriever(
			weaviate_store=self.weaviate_store,
			parent_store=self.parent_store,
			kg_searcher=self.kg_searcher,
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

		Args:
			history: Optional list of role/content dictionaries.

		Returns:
			A cleaned list of role/content dictionaries limited to the latest
			turns that the model should see.
		"""
		if not history:
			return []

		normalized: List[Dict[str, str]] = []
		for item in history:
			role = item.get("role")
			content = item.get("content")
			if role not in {"system", "user", "assistant"}:
				continue
			if not isinstance(content, str) or not content.strip():
				continue
			normalized.append({"role": role, "content": content.strip()})

		return normalized[-5:]

	def _build_entity_texts(self, analysis: Dict[str, Any]) -> List[str]:
		"""Build entity texts for MedCPT Article-Encoder embedding.

		Args:
			analysis: Result dictionary from QueryAnalyzer.

		Returns:
			A list of entity strings enriched with their coarse type labels.
		"""
		entity_texts: List[str] = []

		for disease in analysis.get("diseases", []):
			if isinstance(disease, str) and disease.strip():
				entity_texts.append(f"Disease: {disease.strip()}")

		for symptom in analysis.get("symptoms", []):
			if isinstance(symptom, str) and symptom.strip():
				entity_texts.append(f"EffectPhenotype: {symptom.strip()}")

		for drug in analysis.get("drugs", []):
			if isinstance(drug, str) and drug.strip():
				entity_texts.append(f"Drug: {drug.strip()}")

		return entity_texts

	def _build_canonical_items(self, ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Convert reranked output into the canonical prompt schema.

		Args:
			ranked_results: Unified list from the cross-encoder reranker.

		Returns:
			A list of prompt items with `source_type`, `text`, and scores.
		"""
		canonical: List[Dict[str, Any]] = []

		for item in ranked_results:
			source_type = item.get("source_type", "text_retrieval")
			text = item.get("text", "")
			if not isinstance(text, str) or not text.strip():
				continue

			canonical_item: Dict[str, Any] = {
				"source_type": source_type,
				"text": text,
				"cross_encoder_score": item.get("cross_encoder_score", 0.0),
			}

			if source_type == "text_retrieval":
				canonical_item.update({
					"parent_id": item.get("parent_id"),
					"pmid": item.get("pmid"),
					"title": item.get("title", ""),
					"rrf_score": item.get("rrf_score", 0.0),
				})
			else:
				canonical_item["metadata"] = item.get("metadata", {})

			canonical.append(canonical_item)

		return canonical

	def run(
		self,
		query: str,
		history: Optional[List[Dict[str, str]]] = None,
		conversation_id: Optional[str] = None,
		top_k: int = settings.RETREVAL_TOP_K,
	) -> Dict[str, Any]:
		"""Run the full synchronous RAG pipeline.

		Args:
			query: Raw user question.
			history: Optional conversation history for query rewriting and answer generation.
			conversation_id: Optional conversation identifier to carry through the response.
			top_k: Retrieval depth for vector and BM25 stages.

		Returns:
			A dictionary with the answer, sources, rewritten query, entities,
			intents, and conversation_id.
		"""
		normalized_history = self._normalize_history(history)

		logger.info("[RAGPipeline] Starting query analysis")
		analysis = self.query_analyzer.analyze(query=query, history=normalized_history)
		rewritten_query = analysis.get("rewritten_query", query)
		intents = analysis.get("intents", ["general"])

		query_vector = self.query_embedder.embed_texts(rewritten_query)[0].tolist()
		entity_texts = self._build_entity_texts(analysis)
		entity_article_embeddings: List[List[float]] = []
		if entity_texts:
			entity_article_embeddings = self.entity_embedder.embed_texts(entity_texts).tolist()

		logger.info("[RAGPipeline] Starting parallel retrieval")
		vector_results, bm25_results, kg_results = self.parallel_retriever.retrieve(
			query_text=rewritten_query,
			query_vector=query_vector,
			entity_article_embeddings=entity_article_embeddings,
			intents=intents,
			top_k=top_k,
		)

		logger.info("[RAGPipeline] Running RRF on text retrieval streams")
		rrf_results = self.rrf_manager.rank_fusion(
			vector_results=vector_results,
			bm25_results=bm25_results,
			top_k=top_k,
		)

		logger.info("[RAGPipeline] Running cross-encoder reranking")
		ranked_results = self.cross_encoder_reranker.rerank(
			query=rewritten_query,
			rrf_results=rrf_results,
			kg_text=kg_results,
			top_m=settings.RERANK_TEXT_TOP_M,
			top_n=settings.RERANK_KG_TOP_N,
		)

		logger.info("[RAGPipeline] Merging KG paths and preparing prompt context")
		merged_results = self.kg_merger.merge_top_paths(ranked_results)

		canonical_items = self._build_canonical_items(merged_results)

		if not canonical_items:
			canonical_items = self._build_canonical_items(ranked_results)

		system_prompt, user_prompt = build_prompts(
			query=rewritten_query,
			retrieved_items=canonical_items,
		)

		logger.info("[RAGPipeline] Generating final answer")
		answer = self.llm_generator.generate_answer(
			system_prompt=system_prompt,
			user_prompt=user_prompt,
			history=normalized_history,
		)

		return {
			"answer": answer,
			"sources": canonical_items,
			"conversation_id": conversation_id,
			"rewritten_query": rewritten_query,
			"analysis": analysis,
		}

