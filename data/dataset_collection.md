# Dataset Collection for MedKG-RAG Knowledge Base

This document outlines how the data for the Medical Knowledge Graph (PrimeKG) and the unstructured text corpus (BioASQ) was collected and processed to build the hybrid retrieval database for the MedKG-RAG chatbot.

## 1. Knowledge Graph Data (Neo4j)

The structured component of the knowledge base is built upon **PrimeKG**, a comprehensive, open-source multimodal knowledge graph created by researchers at Harvard.

### 1.1. Collection and Filtering
To ensure maximum domain specificity and optimize query performance, the raw PrimeKG dataset is not imported entirely. It is rigorously filtered to focus exclusively on the **Disease-Drug-Target** triad.

**Process:**
1. **Node Selection:** We retain only nodes belonging to our core target semantic types:
   - `Disease`
   - `Drug`
   - `GeneProtein`
   - `EffectPhenotype` (which encompasses Symptoms and Side Effects)
2. **Edge Selection:** We scan the graph and keep only the relationships (edges) where *both* endpoints belong to the target node types mentioned above.
3. **Relation Mapping:** Medically relevant relationships are mapped to Neo4j edge labels. For example:
   - `indication` → `TREATS` (Drug → Disease)
   - `side effect` → `HAS_SIDE_EFFECT` (Drug → EffectPhenotype)
   - `target` → `TARGETS` (Drug → GeneProtein)
   - `phenotype present` → `PRESENTS` (Disease → EffectPhenotype)

### 1.2. Graph Processing
The filtered subset of nodes and relationships is then imported into **Neo4j**. As part of the ingestion process (`build_kg.py`), each node undergoes text enrichment (e.g., prepending the node type like `"Disease: Type 2 Diabetes"`) and is encoded using the `MedCPT-Article-Encoder`. The resulting 768-dimensional embeddings are stored directly as node properties to enable Stage 1 vector-based anchor lookup.

---

## 2. Unstructured Text Corpus (Weaviate & SQLite)

The unstructured text corpus, which powers both Semantic (Vector) and Lexical (BM25) searches, is constructed dynamically from the articles referenced in our evaluation datasets.

### 2.1. Collection via NCBI E-utilities
Instead of indexing a massive, unfocused dump of PubMed, the corpus is built to guarantee full coverage of the documents required by our datasets (BioASQ Task B and MedAESQA).

**Process:**
1. **PMID Extraction:** We iterate through all QA pairs in the BioASQ and MedAESQA validation and test splits, extracting all explicitly referenced PubMed IDs (PMIDs).
2. **Dynamic Fetching:** For every unique PMID, we query the **NCBI E-utilities API** (PubMed/PMC).
3. **Data Download:** We download the exact Title and Abstract text associated with each PMID.

### 2.2. Processing and Indexing
Once the raw text is collected:
1. **Adaptive Chunking:** The articles are passed through a 3-Tier Parent-Child Chunker (utilizing `SciSpaCy`). This safely splits long medical abstracts into smaller "Child" chunks while preserving the full abstract as the "Parent" context.
2. **Semantic Encoding:** The child chunks are embedded using the `MedCPT-Article-Encoder`.
3. **Dual Database Storage:**
   - **Weaviate:** Stores the child chunks along with their vector embeddings and BM25 text indices for fast parallel retrieval.
   - **SQLite:** Stores the full parent chunks (`parent_chunks.db`). When a child chunk is matched in Weaviate, the system uses its `parent_id` to retrieve the unbroken parent text from SQLite, which is then fed to the LLM for answer generation.
