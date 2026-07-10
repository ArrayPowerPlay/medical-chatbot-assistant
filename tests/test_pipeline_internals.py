import unittest
import asyncio
from typing import Any

from src.pipeline.rag_pipeline import RAGPipeline
from src.reranking.cross_encoder import CrossEncoderReranker


class _DummyRemoteMethod:
    def __init__(self, scores=None, error=None):
        self.scores = scores or []
        self.error = error

    def remote(self, query, passages):
        if self.error is not None:
            raise self.error
        return self.scores


class _DummyModel:
    def __init__(self, scores=None, error=None):
        self.rerank = _DummyRemoteMethod(scores=scores, error=error)


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_rerank_keeps_negative_scores_and_sorts_all_passages(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.is_available = True
        reranker.model = _DummyModel(scores=[-0.2, 1.5, -1.0])  # type: ignore

        rrf_results = [
            {"parent_id": "a", "pmid": "1", "text": "doc a"},
            {"parent_id": "b", "pmid": "2", "text": "doc b"},
            {"parent_id": "c", "pmid": "3", "text": "doc c"},
        ]

        ranked_text, ranked_kg = asyncio.run(reranker.rerank(
            query="test query",
            rrf_results=rrf_results,
            kg_results=[],
            top_m=3,
            top_n=0,
        ))

        self.assertEqual([item["parent_id"] for item in ranked_text], ["b", "a", "c"])
        self.assertEqual([item["cross_encoder_score"] for item in ranked_text], [1.5, -0.2, -1.0])
        self.assertEqual(ranked_kg, [])

    def test_rerank_exception_uses_structured_fallback_for_text_and_kg(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.is_available = True
        reranker.model = _DummyModel(error=RuntimeError("boom"))  # type: ignore[assignment]

        ranked_text, ranked_kg = asyncio.run(reranker.rerank(
            query="test query",
            rrf_results=[
                {"parent_id": "a", "pmid": "1", "text": "doc a"},
                {"parent_id": "b", "pmid": "2", "text": "doc b"},
            ],
            kg_results=[
                {"text": "kg path", "metadata": {"path_id": "kg-1"}},
            ],
            top_m=1,
            top_n=1,
        ))

        self.assertEqual(len(ranked_text), 1)
        self.assertEqual(ranked_text[0]["parent_id"], "a")
        self.assertEqual(ranked_text[0]["source_type"], "text_retrieval")
        self.assertEqual(len(ranked_kg), 1)
        self.assertEqual(ranked_kg[0]["text"], "kg path")
        self.assertEqual(ranked_kg[0]["source_type"], "kg_retrieval")


class RAGPipelineInternalsTests(unittest.TestCase):
    def test_normalize_history_slices_after_filtering(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        history = [
            {"role": "user", "content": "  first  "},
            {"role": "invalid", "content": "skip me"},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": " second "},
        ]

        normalized = pipeline._normalize_history(history, max_messages=1)

        self.assertEqual(normalized, [{"role": "assistant", "content": "second"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
