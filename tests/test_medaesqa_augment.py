import unittest

from src.dataset_builder.augment_corpus_with_medaesqa import collect_gold_pmids
from src.dataset_builder.preprocess_bioasq_taskB import (
    _parse_pmc_html,
    _parse_pmc_xml,
    _parse_pubmed_xml,
)


class MedAESQAAugmentTests(unittest.TestCase):
    def test_collect_gold_pmids_ignores_machine_citations(self):
        dataset = [
            {
                "expert_curated_answer": "Gold answer [123, 456].",
                "machine_generated_answers": {
                    "M1": {
                        "answer_sentences": [
                            {
                                "citation_assessment": [
                                    {"cited_pmid": "789"},
                                    {"cited_pmid": "5"},
                                ]
                            }
                        ]
                    }
                },
            }
        ]

        self.assertEqual(collect_gold_pmids(dataset), {"123", "456"})

    def test_parse_pubmed_xml_uses_itertext_and_extracts_pmcid(self):
        xml_content = """
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>111</PMID>
              <Article>
                <ArticleTitle>Deep <i>Nested</i> Title</ArticleTitle>
                <Abstract>
                  <AbstractText Label="BACKGROUND">First <b>section</b>.</AbstractText>
                  <AbstractText Label="METHODS">Second section.</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
            <PubmedData>
              <ArticleIdList>
                <ArticleId IdType="pmc">PMC12345</ArticleId>
              </ArticleIdList>
            </PubmedData>
          </PubmedArticle>
        </PubmedArticleSet>
        """

        records = _parse_pubmed_xml(xml_content)
        self.assertIn("111", records)
        self.assertEqual(records["111"]["title"], "Deep Nested Title")
        self.assertIn("BACKGROUND: First section.", records["111"]["abstractText"])
        self.assertIn("METHODS: Second section.", records["111"]["abstractText"])
        self.assertEqual(records["111"]["pmcid"], "PMC12345")
        self.assertEqual(records["111"]["content_source"], "pubmed_abstract")

    def test_parse_pubmed_xml_supports_pubmed_book_article(self):
        xml_content = """
        <PubmedArticleSet>
          <PubmedBookArticle>
            <BookDocument>
              <PMID Version="1">222</PMID>
              <ArticleTitle book="statpearls">Book Excerpt Title</ArticleTitle>
              <Abstract>
                <AbstractText>Excerpt text from book-style PubMed record.</AbstractText>
              </Abstract>
            </BookDocument>
          </PubmedBookArticle>
        </PubmedArticleSet>
        """

        records = _parse_pubmed_xml(xml_content)
        self.assertIn("222", records)
        self.assertEqual(records["222"]["title"], "Book Excerpt Title")
        self.assertEqual(
            records["222"]["abstractText"],
            "Excerpt text from book-style PubMed record.",
        )
        self.assertEqual(records["222"]["record_type"], "PubmedBookArticle")

    def test_parse_pmc_xml_returns_plain_text_fallback(self):
        xml_content = """
        <article>
          <front>
            <article-meta>
              <title-group>
                <article-title>PMC Title</article-title>
              </title-group>
              <abstract>
                <p>Abstract paragraph.</p>
              </abstract>
            </article-meta>
          </front>
          <body>
            <sec>
              <p>Body paragraph one.</p>
              <p>Body paragraph two.</p>
            </sec>
          </body>
        </article>
        """

        record = _parse_pmc_xml(xml_content, pmid="222", pmcid="PMC222")
        self.assertIsNotNone(record)
        if record:
            self.assertEqual(record["pmid"], "222")
            self.assertEqual(record["pmcid"], "PMC222")
            self.assertEqual(record["title"], "PMC Title")
            self.assertIn("Abstract paragraph.", record["abstractText"])
            self.assertIn("Body paragraph one.", record["abstractText"])
            self.assertEqual(record["content_source"], "pmc_fulltext_fallback")

    def test_parse_pmc_html_returns_plain_text_fallback(self):
        html_content = """
        <html>
          <body>
            <h1 class="content-title">PMC HTML Title</h1>
            <section aria-label="Article content">
              <section class="body main-article-body">
                <section id="sec1">
                  <p>Paragraph one.</p>
                </section>
                <section id="sec2">
                  <p>Paragraph two with <strong>markup</strong>.</p>
                </section>
              </section>
            </section>
          </body>
        </html>
        """

        record = _parse_pmc_html(html_content, pmid="333", pmcid="PMC333")
        self.assertIsNotNone(record)
        if record:
            self.assertEqual(record["title"], "PMC HTML Title")
            self.assertIn("Paragraph one.", record["abstractText"])
            self.assertIn("Paragraph two with markup.", record["abstractText"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
