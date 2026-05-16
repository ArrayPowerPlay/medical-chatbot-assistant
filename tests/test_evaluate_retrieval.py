import io
import json
import logging
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_retrieval as er


class _DummyResource:
    def __init__(self):
        self.closed = False
        self.temperature = 0.0

    def close(self):
        self.closed = True


class EvaluateRetrievalTests(unittest.TestCase):
    def test_geometric_mean_average_precision_handles_zero_ap(self):
        metrics_list = [
            {"average_precision_at_5": 1.0},
            {"average_precision_at_5": 0.25},
            {"average_precision_at_5": 0.0},
        ]

        gmap = er.geometric_mean_average_precision(metrics_list, "average_precision_at_5")
        expected = round(
            math.exp(
                (
                    math.log(1.0 + er.GMAP_EPSILON)
                    + math.log(0.25 + er.GMAP_EPSILON)
                    + math.log(0.0 + er.GMAP_EPSILON)
                ) / 3
            ),
            4,
        )

        self.assertEqual(gmap, expected)

    def test_compute_document_metrics_uses_top_k_correctly(self):
        retrieved_pmids = ["p1", "x1", "x2", "x3", "x4", "p2"]
        gold_pmids = {"p1", "p2"}

        metrics = er.compute_document_metrics(retrieved_pmids, gold_pmids)

        self.assertEqual(metrics["precision_at_5"], 0.2)
        self.assertEqual(metrics["recall_at_5"], 0.5)
        self.assertEqual(metrics["f1_at_5"], 0.2857)
        self.assertEqual(metrics["average_precision_at_5"], 0.5)
        self.assertEqual(metrics["reciprocal_rank"], 1.0)

    def test_compute_snippet_metrics_matches_substrings_per_pmid(self):
        retrieved_items = [
            {"pmid": "p1", "text": "alpha relevant snippet beta"},
            {"pmid": "p2", "text": "gamma unrelated"},
            {"pmid": "p3", "text": "delta relevant snippet again"},
        ]
        gold_snippets = [
            {"pmid": "p1", "text": "relevant snippet"},
            {"pmid": "p3", "text": "relevant snippet"},
        ]

        metrics = er.compute_snippet_metrics(retrieved_items, gold_snippets, {"p1", "p3"})

        self.assertEqual(metrics["snippet_recall_at_5"], 1.0)
        self.assertEqual(metrics["snippet_precision_at_5"], 0.4)
        self.assertEqual(metrics["snippet_f1_at_5"], 0.5714)

    def test_build_summary_includes_gmap_and_new_config_keys(self):
        all_doc_metrics = [
            {
                "precision_at_5": 0.5,
                "recall_at_5": 0.25,
                "f1_at_5": 0.3333,
                "average_precision_at_5": 0.5,
                "precision_at_10": 0.4,
                "recall_at_10": 0.4,
                "f1_at_10": 0.4,
                "average_precision_at_10": 0.4,
                "precision_at_20": 0.2,
                "recall_at_20": 0.5,
                "f1_at_20": 0.2857,
                "average_precision_at_20": 0.2,
                "reciprocal_rank": 1.0,
            },
            {
                "precision_at_5": 0.0,
                "recall_at_5": 0.0,
                "f1_at_5": 0.0,
                "average_precision_at_5": 0.0,
                "precision_at_10": 0.1,
                "recall_at_10": 0.2,
                "f1_at_10": 0.1333,
                "average_precision_at_10": 0.1,
                "precision_at_20": 0.1,
                "recall_at_20": 0.4,
                "f1_at_20": 0.16,
                "average_precision_at_20": 0.05,
                "reciprocal_rank": 0.0,
            },
        ]
        all_snippet_metrics = [
            {
                "snippet_recall_at_5": 0.5,
                "snippet_precision_at_5": 0.4,
                "snippet_f1_at_5": 0.4444,
                "snippet_recall_at_10": 0.6,
                "snippet_precision_at_10": 0.3,
                "snippet_f1_at_10": 0.4,
                "snippet_recall_at_20": 0.7,
                "snippet_precision_at_20": 0.2,
                "snippet_f1_at_20": 0.3111,
            }
        ]

        summary = er._build_summary(all_doc_metrics, all_snippet_metrics, 2, 0)

        self.assertIn("GMAP@5", summary["document_metrics"])
        self.assertIn("GMAP@10", summary["document_metrics"])
        self.assertIn("GMAP@20", summary["document_metrics"])
        self.assertIn("vector_top_k", summary["config"])
        self.assertIn("keyword_top_k", summary["config"])
        self.assertNotIn("retrieval_top_k", summary["config"])

    def test_build_arg_parser_rejects_removed_question_id(self):
        parser = er.build_arg_parser()
        args = parser.parse_args(["--limit", "3"])
        self.assertEqual(args.limit, 3)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--question-id", "q1"])

    def test_evaluate_logs_pipeline_errors_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            val_path = tmp_path / "val_bioasq.jsonl"
            output_dir = tmp_path / "eval_results"

            questions = [
                {
                    "id": "q-fail",
                    "body": "broken question",
                    "relevant_pmid": ["1"],
                    "snippets": [{"pmid": "1", "text": "gold"}],
                },
                {
                    "id": "q-ok",
                    "body": "working question",
                    "relevant_pmid": ["1"],
                    "snippets": [{"pmid": "1", "text": "gold"}],
                },
            ]
            with val_path.open("w", encoding="utf-8") as f:
                for item in questions:
                    f.write(json.dumps(item) + "\n")

            query_analyzer = _DummyResource()
            query_embedder = _DummyResource()
            weaviate_store = _DummyResource()
            parent_store = _DummyResource()
            rrf_manager = object()
            cross_encoder = object()

            def fake_run_retrieval_pipeline(
                query,
                query_analyzer,
                query_embedder,
                weaviate_store,
                parent_store,
                rrf_manager,
                cross_encoder,
                debug_label=None,
            ):
                if query == "broken question":
                    raise ZeroDivisionError("float division by zero")

                return (
                    [
                        {
                            "parent_id": "parent-1",
                            "pmid": "1",
                            "title": "doc",
                            "text": "prefix gold suffix",
                            "cross_encoder_score": 0.9,
                        }
                    ],
                    "rewritten working question",
                )

            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            handler.setLevel(logging.ERROR)
            er.logger.addHandler(handler)

            try:
                with patch.object(er, "VAL_PATH", val_path), \
                     patch.object(er, "OUTPUT_DIR", output_dir), \
                     patch.object(er, "_print_summary"), \
                     patch.object(er, "QueryAnalyzer", return_value=query_analyzer), \
                     patch.object(er, "MedCPTEmbedder", return_value=query_embedder), \
                     patch.object(er, "WeaviateChildStore", return_value=weaviate_store), \
                     patch.object(er, "ParentStore", return_value=parent_store), \
                     patch.object(er, "RRFManager", return_value=rrf_manager), \
                     patch.object(er, "CrossEncoderReranker", return_value=cross_encoder), \
                     patch.object(er, "run_retrieval_pipeline", side_effect=fake_run_retrieval_pipeline):
                    er.evaluate(limit=2)
            finally:
                er.logger.removeHandler(handler)

            error_text = log_stream.getvalue()
            print("\nCaptured evaluate_retrieval error log:\n")
            print(error_text)

            self.assertIn("Failed on question q-fail:", error_text)
            self.assertIn("ZeroDivisionError: float division by zero", error_text)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["config"]["failed_questions"], 1)
            self.assertEqual(summary["config"]["evaluated_questions"], 1)
            self.assertIn("vector_top_k", summary["config"])
            self.assertIn("keyword_top_k", summary["config"])
            self.assertNotIn("retrieval_top_k", summary["config"])
            self.assertIn("GMAP@5", summary["document_metrics"])
            self.assertIn("GMAP@10", summary["document_metrics"])
            self.assertIn("GMAP@20", summary["document_metrics"])

            detail_lines = (
                output_dir / "detail.jsonl"
            ).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(detail_lines), 1)

            detail_record = json.loads(detail_lines[0])
            self.assertEqual(detail_record["question_id"], "q-ok")
            self.assertEqual(detail_record["rewritten_query"], "rewritten working question")

            self.assertTrue(query_analyzer.closed)
            self.assertTrue(query_embedder.closed)
            self.assertTrue(weaviate_store.closed)
            self.assertTrue(parent_store.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
