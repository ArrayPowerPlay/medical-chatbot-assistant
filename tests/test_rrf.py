import unittest

from src.reranking.rrf import RRFManager


class RRFManagerTests(unittest.TestCase):
    def test_rank_fusion_merges_and_sorts_by_rrf_score(self):
        vector_results = [
            {"parent_id": "a", "pmid": "1", "text": "doc a", "score": 0.9},
            {"parent_id": "b", "pmid": "2", "text": "doc b", "score": 0.8},
        ]
        bm25_results = [
            {"parent_id": "b", "pmid": "2", "text": "doc b", "score": 12.0},
            {"parent_id": "c", "pmid": "3", "text": "doc c", "score": 11.0},
        ]

        fused = RRFManager(k=60).rank_fusion(vector_results, bm25_results, top_k=10)

        self.assertEqual([item["parent_id"] for item in fused], ["b", "a", "c"])
        self.assertEqual(fused[0]["source_type"], "text_retrieval")
        self.assertIn("rrf_score", fused[0])
        self.assertNotIn("score", fused[0])

    def test_negative_k_raises_clear_error_instead_of_dividing_by_zero(self):
        with self.assertRaisesRegex(ValueError, r"RRF parameter k must be >= 0, got -1"):
            RRFManager(k=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
