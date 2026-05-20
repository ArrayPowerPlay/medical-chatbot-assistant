"""Adaptive parent-child chunking utilities for PubMed title and abstract records."""

import spacy
from typing import List, Dict
from config.settings import settings

# Load SciSpaCy and enable only the sentencizer for speed acceleration
try:
    nlp = spacy.load("en_core_sci_sm", disable=["tagger", "parser", "ner", "lemmatizer", "textcat"])
    nlp.add_pipe("sentencizer")
except Exception:
    raise ImportError("Please install: pip install scispacy en_core_sci_sm")


class AdaptiveChunker:
    """
    Implement the 3-Tier Adaptive Parent-Child Chunking strategy.
    Tier 1: <= 500 chars (parent chunk = child chunk = full)
    Tier 2: <= 2000 chars (parent chunk = full, child chunk = 500 chars)
    Tier 3: > 2000 chars (parent chunk = 1500 chars - 256 chars overlap, child chunk = 500 chars)
    """
    @staticmethod
    def _split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split large text into parent chunks using SciSpaCy sentences with an exact character overlap."""
        doc = nlp(text)
        # doc.sents = list of sentences after NLP processing and remove overly short sentences
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

        chunks = []
        current_chunk = ""
        current_sentences = []     # Use for building overlapping chunks

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size or current_chunk == "":
                current_chunk += (" "if current_chunk else "") + sentence
                current_sentences.append(sentence)
            else:
                chunks.append(current_chunk)
                # Calculate the overlap by backtracking through current sentence
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_sentences):
                    if len(overlap_text) + len(s) <= overlap:
                        overlap_text = s + (" " if overlap_text else "") + overlap_text
                        overlap_sentences.insert(0, s)
                    else:
                        break
                # Start a new chunk with the overlap
                current_chunk = overlap_text + (" " if overlap_text else "") + sentence
                current_sentences = overlap_sentences + [sentence]

        # Add the remaining chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks
    
    @staticmethod
    def _split_child_chunks(text: str, title: str, chunk_size: int) -> List[str]:
        """Splits parent texts into child chunks and injects the title."""
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size or current_chunk == "":
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk.startswith(f"Title: {title}"):
                    chunks.append(current_chunk)
                else:
                    chunks.append(f"Title: {title}.\nContent: {current_chunk}")
                current_chunk = sentence

        if current_chunk:
            if current_chunk.startswith(f"Title: {title}"):
                chunks.append(current_chunk)
            else:
                chunks.append(f"Title: {title}.\nContent: {current_chunk}")
        return chunks
    
    @staticmethod
    def process_article(article_id: str, title: str, abstract: str) -> Dict:
        """Main entry point for processing a single BioASQ article."""
        full_text = f"Title: {title}\nAbstract: {abstract}".strip()
        length = len(full_text)

        parents = []
        children_mapping = {}

        if length <= settings.TIER1_MAX_LEN:
            p_id = f"{article_id}_p0"
            parents.append({"parent_id": p_id, "pmid": article_id, "title": title, "text": full_text})
            children_mapping[p_id] = [full_text]

        elif length <= settings.TIER2_MAX_LEN:
            p_id = f"{article_id}_p0"
            parents.append({"parent_id": p_id, "pmid": article_id, "title": title, "text": full_text})
            children_mapping[p_id] = AdaptiveChunker._split_child_chunks(
                text=full_text,
                title=title,
                chunk_size=settings.CHILD_CHUNK_SIZE
            )

        else:
            parent_texts = AdaptiveChunker._split_text_with_overlap(
                text=full_text,
                chunk_size=settings.PARENT_CHUNK_SIZE,
                overlap=settings.PARENT_CHUNK_OVERLAP
            )
            for i, p_text in enumerate(parent_texts):
                p_id = f"{article_id}_p{i}"
                parents.append({"parent_id": p_id, "pmid": article_id, "title": title, "text": p_text})
                children_mapping[p_id] = AdaptiveChunker._split_child_chunks(
                    text=p_text,
                    title=title,
                    chunk_size=settings.CHILD_CHUNK_SIZE
                )

        return {"parents": parents, "children": children_mapping}
