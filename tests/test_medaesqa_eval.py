import unittest
from scripts.evaluation.shared.generation_medaesqa_common import (
    extract_pmids_from_text,
    compute_citation_metrics,
)
from src.generation.prompt_builder import build_prompts


class MedAESQAEvalTests(unittest.TestCase):
    def test_extract_pmids_from_text(self):
        text = "This is a statement [12345, 67890] and another [111]."
        self.assertEqual(extract_pmids_from_text(text), {"12345", "67890", "111"})

    def test_extract_pmids_from_empty_text(self):
        self.assertEqual(extract_pmids_from_text(""), set())
        self.assertEqual(extract_pmids_from_text(None), set())

    def test_compute_citation_metrics(self):
        prediction = "Treatment is good [12345, 111]."
        gold_pmids = {"12345", "67890"}
        metrics = compute_citation_metrics(prediction, gold_pmids)
        self.assertEqual(metrics["citation_precision"], 0.5)
        self.assertEqual(metrics["citation_recall"], 0.5)

    def test_compute_citation_metrics_no_citations(self):
        prediction = "Treatment is good."
        gold_pmids = {"12345", "67890"}
        metrics = compute_citation_metrics(prediction, gold_pmids)
        self.assertEqual(metrics["citation_precision"], 0.0)
        self.assertEqual(metrics["citation_recall"], 0.0)

    def test_build_prompts_with_and_without_citations(self):
        retrieved_items = [
            {
                "source_type": "text_retrieval",
                "text": "First passage info.",
                "pmid": "9999",
            }
        ]
        
        # Test with citations enabled
        sys_p_enabled, user_p_enabled = build_prompts(
            query="test query",
            retrieved_items=retrieved_items,
            use_head_tail_placement=False,
            use_citations=True
        )
        self.assertIn("PMID", user_p_enabled)
        self.assertIn("9999", user_p_enabled)
        self.assertIn("MUST cite", sys_p_enabled)

        # Test with citations disabled
        sys_p_disabled, user_p_disabled = build_prompts(
            query="test query",
            retrieved_items=retrieved_items,
            use_head_tail_placement=False,
            use_citations=False
        )
        self.assertNotIn("PMID", user_p_disabled)
        self.assertNotIn("9999", user_p_disabled)
        self.assertNotIn("MUST cite", sys_p_disabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
