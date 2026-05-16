# Implementation Plan: Paper-Ready BioASQ-Like Retrieval Metrics

## Summary

Goal: upgrade the current retrieval evaluator from an internal debugging tool into a
paper-ready evaluator that reports both:

- custom diagnostic metrics currently used in the repo
- BioASQ official-like metrics for document and snippet retrieval

The implementation should preserve the current retrieval pipeline while adding:

- enriched gold snippet metadata with section and span offsets
- child-level snippet candidate ranking
- official-like document metrics at top-10
- official-like snippet overlap metrics at top-10
- dual reporting for the full validation set and the title/abstract-only subset

## Key Changes

### 1. Enrich evaluation datasets

Current `val_bioasq.jsonl` / `test_bioasq.jsonl` store only:

- `snippet.text`
- `snippet.pmid`

To support official-like snippet scoring, extend each gold snippet to preserve
BioASQ-style provenance fields:

```json
{
  "text": "...",
  "pmid": "12345",
  "document": "http://www.ncbi.nlm.nih.gov/pubmed/12345",
  "beginSection": "title|abstract|sections.0|...",
  "endSection": "title|abstract|sections.0|...",
  "offsetInBeginSection": 123,
  "offsetInEndSection": 245
}
```

Also add per-question flags:

- `has_non_title_abstract_gold_snippet`
- `all_gold_snippets_in_title_abstract`

These flags will support two snippet reports:

- full validation set
- title/abstract-only subset

### 2. Preserve span-safe child metadata

The current text retrieval pipeline indexes child chunks for retrieval and aggregates
them to parents for document ranking. For official-like snippet evaluation, each child
chunk must carry two text views:

- `indexed_text`
  - used by vector/BM25 retrieval
  - may keep title injection if desired for retrieval quality
- `span_text`
  - raw span-bearing text used for offset-aware snippet prediction
  - must not contain synthetic prefixes that would corrupt offsets

Each child chunk should also carry section-local offsets:

```json
{
  "child_id": "...",
  "parent_id": "...",
  "pmid": "...",
  "indexed_text": "...",
  "span_text": "...",
  "begin_section": "title|abstract",
  "end_section": "title|abstract",
  "offset_begin": 0,
  "offset_end": 187
}
```

V1 scope:

- snippet prediction supports only sections actually present in the corpus: `title` and `abstract`
- no attempt is made to hallucinate `sections.*` full-text spans

### 3. Add a snippet-candidate retrieval path

Current evaluation infers snippet quality indirectly from:

- ranked parent chunks
- substring containment of gold snippets inside parent text

That is not sufficient for paper-ready official-like snippet evaluation.

Add a separate snippet path:

- retrieve raw child hits from vector search and BM25 before parent aggregation
- fuse child hits with RRF
- rerank child hits with the same cross-encoder
- convert top child hits into predicted snippet records with:
  - `document`
  - `pmid`
  - `text`
  - `beginSection`
  - `endSection`
  - `offsetInBeginSection`
  - `offsetInEndSection`
  - `score`

Deduplicate snippet candidates by:

- `(pmid, beginSection, endSection, offsetInBeginSection, offsetInEndSection)`

Keep top-10 predicted snippets for official-like scoring.

### 4. Keep document and snippet evaluation separate

#### Document path

Keep the current parent-oriented pipeline for document evaluation:

- QueryAnalyzer
- vector child retrieval + BM25 child retrieval
- aggregate child to parent
- parent-level RRF
- parent-level cross-encoder rerank
- top-10 documents for official-like metrics

#### Snippet path

Use child chunks as snippet candidates:

- vector child retrieval + BM25 child retrieval
- child-level RRF
- child-level cross-encoder rerank
- top-10 snippets for official-like metrics

This avoids pretending that a parent chunk is itself a snippet.

### 5. Add official-like document metrics

For top-10 predicted documents, compute:

- Mean Precision
- Recall
- F-Measure
- MAP
- GMAP

Use macro-averaging across questions.

### 6. Add official-like snippet overlap metrics

For top-10 predicted snippets, compute overlap-based metrics using:

- PMID
- section
- character offsets

The overlap unit is the set of covered character positions scoped by:

- document / PMID
- section

#### Official snippet scoring logic in detail

BioASQ does **not** treat snippets as exact string labels. A snippet is defined by:

- the article it comes from
- the section it belongs to
- the offset of its first character
- the offset of its last character

Operationally, each snippet is converted into a set of character-level units:

- one unit per covered character position
- scoped by `(document, section, offset)`

For a question:

- let `G` be the union of all gold snippet character units
- let `S` be the union of all predicted snippet character units

Then official-like snippet precision and recall are:

- `Psnip = |S ∩ G| / |S|`
- `Rsnip = |S ∩ G| / |G|`

Interpretation:

- `Psnip` measures how much of the returned snippet mass overlaps gold
- `Rsnip` measures how much of the gold snippet mass is covered by the returned snippets
- partial overlaps receive partial credit automatically because overlap is counted in characters, not in all-or-nothing snippet IDs

Per-question snippet F-measure:

- `Fsnip = 2 * Psnip * Rsnip / (Psnip + Rsnip)`
- if both `Psnip` and `Rsnip` are `0`, define `Fsnip = 0`

#### Ranked snippet evaluation

For ranked snippet lists, the score at rank `r` is computed on the prefix of the first `r` returned snippets.

For prefix `L[:r]`:

- compute `Psnip(r)` using only the first `r` predicted snippets
- define `rel(r) = 1` if the `r`-th predicted snippet has **non-zero** overlap with at least one gold snippet of the same question
- define `rel(r) = 0` otherwise

Average Precision for snippets:

```text
AP_snip = ( Σ_r Psnip(r) * rel(r) ) / |L_R|
```

Where:

- `|L_R|` is the number of relevant snippet items for the question under the ranked-list view
- in the BioASQ-style implementation for snippets, a returned snippet is relevant if it has non-zero overlap with at least one gold snippet

In practice for this repo, use the BioASQ-style interpretation:

- `P(r)` in AP is `Psnip(r)` on the prefix
- `rel(r)=1` iff overlap with gold is non-zero

Then aggregate across questions:

- `MAP_snip = mean(AP_snip_i)`
- `GMAP_snip = exp( mean( log(AP_snip_i + eps) ) )`

With:

- `eps = 1e-6` by default
- `eps` explicitly documented in the summary output

#### How this maps to the current corpus

The current corpus contains PubMed `title + abstract`, not full text.
Raw BioASQ snippets may still be labeled as `sections.0`, `sections.1`, etc.

For this repo, use the following mapping policy:

- if a gold snippet text can be found inside the stored `abstractText`, create an `abstract-proxy` span in the corpus text and score it normally
- if a gold snippet text can be found inside the stored `title`, create a `title-proxy` span and score it normally
- if a gold snippet cannot be located in either title or abstract, mark it as `unmappable_to_corpus`

Observed on the current validation split:

- 158 gold snippets are labeled `sections.*`
- 154 of them are exact substrings of the stored `abstractText`
- 4 are not exact matches and should be treated as `unmappable_to_corpus` until a normalization rule is explicitly added

#### Reporting policy

Because the corpus is title+abstract only, report snippet official-like metrics in two views:

- `snippets_full_set`
  - all validation questions
  - unmappable gold snippets remain counted in gold mass
- `snippets_mappable_subset`
  - only questions whose gold snippets are fully mappable to the current corpus representation

This keeps the paper honest while still allowing a fairer score for the retrievable subset.

Per-question snippet metrics:

- `Psnip = |predicted_chars ∩ gold_chars| / |predicted_chars|`
- `Rsnip = |predicted_chars ∩ gold_chars| / |gold_chars|`
- `Fsnip = 2PR / (P + R)`

Aggregate:

- Mean Precision
- Recall
- F-Measure
- MAP
- GMAP

Report snippet metrics twice:

- `snippets_full_set`
- `snippets_title_abstract_only_subset`

### 7. Extend evaluator CLI and outputs

Extend `scripts/evaluate_retrieval.py` with:

- `--metric-mode custom|official|both`
- `--official-top-k 10`
- `--snippet-scope full|subset|both`

Keep:

- `--limit`
- `--question-id`

Update outputs:

- `summary.json`
- `detail.jsonl`

Suggested `summary.json` structure:

```json
{
  "config": {...},
  "custom_metrics": {...},
  "official_like": {
    "documents": {...},
    "snippets_full_set": {...},
    "snippets_title_abstract_subset": {...},
    "counts": {
      "questions_total": 500,
      "questions_snippet_subset": 487,
      "questions_with_non_title_abstract_gold": 13
    }
  }
}
```

Per-question `detail.jsonl` should add:

- `predicted_documents`
- `predicted_snippets`
- `gold_snippets` with offsets
- `official_doc_metrics`
- `official_snippet_metrics`
- `is_title_abstract_only_question`

## Test Plan

### Unit tests

- document official-like precision / recall / F / MAP / GMAP
- snippet overlap scoring with:
  - full overlap
  - partial overlap
  - no overlap
  - duplicate predictions
  - same PMID but different section

### Dataset enrichment tests

- ensure snippet provenance fields are preserved from `training10b.json`
- ensure subset flags are correct for `title/abstract` vs `sections.*`

### Retrieval-to-snippet conversion tests

- child metadata converts to predicted snippet records correctly
- deduplication works
- top-10 clipping works

### Regression tests

- `--metric-mode custom` preserves current evaluator behavior
- evaluator still logs traceback and continues when a question fails
- official-like mode fails fast with a clear error if offset fields are missing

## Assumptions

- both custom and official-like metrics should remain in the repo
- offsets are precomputed into val/test datasets, not generated lazily at eval time
- the current corpus is still title+abstract only
- snippet official-like reporting must therefore distinguish:
  - full validation set
  - title/abstract-only subset
- V1 uses child chunks as snippet candidates rather than adding a separate span extractor
- official-like cap is top-10 for both documents and snippets
