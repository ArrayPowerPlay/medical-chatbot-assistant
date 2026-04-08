from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple


class ParentChildChunker:
    """Split articles into parent chunks (large, for contexts) and child chunks
    (small, for vector search). Each child references its parent via 'parent_id'"""
    def __init__(
        self,
        parent_chunk_size: int = 1200,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 256,
        child_chunk_overlap: int = 64
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap
        )

    def chunk_article(self, article: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Split one article into many parent and chilren chunks.
        
        Return:
            (parent_chunks, child_chunks)
            - parent_chunks: list of {"parent_id", "pmid", "title", "text"}
            - child_chunks: list of {"parent_id", "pmid", "text"}
        """
        pmid = article.get("pmid", "unknown")
        title = article.get("title", "")
        abstractText = article.get("abstractText", "")
        full_text = f"{title} {abstractText}".strip()   # Chunk based on title + abstract

        if not full_text:
            return [], []
        
        # Split 'full_text' into parent chunks
        parent_texts = self.parent_splitter.split_text(full_text)
        
        parent_chunks = []
        child_chunks = []

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = f"{pmid}_{p_idx}"
            parent_chunks.append({
                "parent_id": parent_id,
                "pmid": pmid,
                "title": title,
                "text": parent_text,
            })

            # Split each parent into smaller child chunks
            child_texts = self.child_splitter.split_text(parent_text)

            for c_idx, child_text in enumerate(child_texts):
                child_chunks.append({
                    "parent_id": parent_id,
                    "pmid": pmid,
                    "text": child_text,
                })

        return parent_chunks, child_chunks
        



        