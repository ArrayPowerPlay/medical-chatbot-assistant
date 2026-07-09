"""
Neo4j Client - Neo4j connection & Inference-time query helpers

Provides:
    - Connection lifecycle (connect, close, health check)
    - 2-hop sub-graph retrieval for KG search at inference time:

        Stage 1: Semantic anchor node search  (Article-Encoder space)
        + For each extracted entity, encode with MedCPT-Article-Encoder and query
        the 'medcpt_node_embeddings' vector index. Anchor sets from every entity 
        are merged and deduplicated.

        Stage 2a: 1-hop neighbour retrieval   (MedCPT cross-space)
        + From anchors, traverse only edge types allowed by query intent.
        Rank neighbours by cosine_sim(rewritten_query_vec, neighbour.embedding_medcpt).
        Keep top-M nodes per anchor.

        Stage 2b: 2-hop neighbour retrieval   (MedCPT cross-space)
        + From 1-hop nodes, traverse intent-allowed edge types.
        Rank by cosine_sim(rewritten_query_vec, neighbour.embedding_medcpt).
        Keep top-N nodes per 1-hop node, hard cap 50 triples total.

Design — two distinct vectors at inference
------------------------------------------
    entity_article_embeddings : [Article-Encoder(e) for e in entities]
        Same encoder space as stored node embeddings
        => Reliable same-space cosine comparison for entity-to-node lookup (Stage 1).

    rewritten_query_vec : Query-Encoder(rewritten_query)
        Standard MedCPT asymmetric comparison: Query Encoder and Article Encoder
        =>  Ranks neighbours by relevance to the full question intent (Stage 2).
"""

import numpy as np
from neo4j import AsyncGraphDatabase
from config.logging_config import logger
from config.settings import settings
from src.kg.schema import INTENT_EDGE_FILTER
from collections import defaultdict
from typing import Any
from src.interfaces.kg import IKGSearcher


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors"""
    v_a = np.array(a, dtype=np.float32)
    v_b = np.array(b, dtype=np.float32)
    norm_mul = np.linalg.norm(v_a) * np.linalg.norm(v_b)
    return float(np.dot(v_a, v_b) / norm_mul) if norm_mul > 0 else 0.0


class Neo4jClient(IKGSearcher):
    """Manages an async Neo4j driver connection and exposes the 2-stage MedCPT
    semantic search + 2-hop traversal retrieval pipeline."""

    def __init__(self):
        logger.info(f"Connecting to Neo4j at {settings.NEO4J_URL}...")
        try:
            # Create driver to connect to Neo4j 
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URL,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            logger.info("Neo4j driver initialized (async).")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    async def close(self):
        """Close the driver"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    # ------------------------------------------------------------------
    # Stage 1 — Anchor search (Article-Encoder space, A-E vs A-E)
    # ------------------------------------------------------------------

    async def _search_anchors_for_entity(
        self,
        entity_article_emb: list[float],
        top_k: int
    ) -> list[str]:
        """Query 'medcpt_node_embeddings' index with a single entity embedding.
        Both the index and the query vector are in Article-Encoder space,
        so cosine comparison is same-space and reliable.

        Args:
            entity_article_emb: Article-Encoder entity string vector.
            top_k: Max anchor nodes to retrieve for this entity.

        Returns:
            List of node names ordered by descending cosine similarity.
        """
        cypher = """
        CALL db.index.vector.queryNodes('medcpt_node_embeddings', $k, $embedding)
        YIELD node, score
        RETURN node.name AS name, score
        ORDER BY score DESC
        """
        try:
            # A temporary session is opened from the driver to execute the query
            async with self.driver.session() as session:          # type: ignore
                result = await session.run(cypher, k=top_k, embedding=entity_article_emb)
                records = await result.data()
                return [r["name"] for r in records if r.get("name")]
        except Exception as e:
            logger.error(f"[Stage 1]: Vector search failed: {e}")
            return []

    async def _find_anchors(
        self,
        entity_article_embeddings: list[list[float]],
        top_k: int
    ) -> list[str]:
        """Run Stage 1 anchor search for all extracted entities.
        Merges results across entities via union + deduplication.

        Args:
            entity_article_embeddings: One Article-Encoder vector per entity.
            top_k: Max anchors per entity.

        Returns:
            Deduplicated list of anchor node names.
        """
        import asyncio
        seen: set[str] = set()
        anchors: list[str] = []
        
        tasks = [self._search_anchors_for_entity(emb, top_k) for emb in entity_article_embeddings]
        results = await asyncio.gather(*tasks)

        for i, candidates in enumerate(results):
            new = [c for c in candidates if c not in seen]
            seen.update(new)
            anchors.extend(new)
            logger.debug(f"[Stage 1]: Entity {i} => Anchors: {candidates}")
            
        logger.info(f"[Stage 1]: Merged anchors ({len(anchors)} unique anchors): {anchors}")
        return anchors

    # ------------------------------------------------------------------
    # Stage 2 — Neighbour retrieval (MedCPT cross-space: Q-E vs A-E)
    # ------------------------------------------------------------------

    async def _get_neighbors(
        self,
        source_names: list[str],
        rewritten_query_vec: list[float],
        allowed_edges: set[str] | None,
        top_n_per_node: int
    ) -> list[dict]:
        """Retrieve and rank neighbours of source nodes.

        Ranking uses cosine_sim(rewritten_query_vec, neighbour.embedding_medcpt):
          - rewritten_query_vec : Query-Encoder(rewritten_query)  → Q-E space
          - embedding_medcpt    : Article-Encoder(node_text)      → A-E space
        This cross-space comparison is the standard MedCPT asymmetric design:

        Args:
            source_names:        Node names to expand from.
            rewritten_query_vec: Query-Encoder(rewritten_query) vector.
            allowed_edges:       Intent-filtered edge types (None = all edges).
            top_n_per_node:      Max neighbours to keep per source node.

        Returns:
            Flat list of {n1, r1, n2, sim} dicts, top_n_per_node per source.
        """
        if not source_names: return []

        if allowed_edges:
            cypher = """
            MATCH (n)-[r]-(m)
            WHERE n.name IN $names AND type(r) IN $edge_types
            RETURN labels(n) AS n1_labels, n.name AS n1, type(r) AS r1, 
                   labels(m) AS n2_labels, m.name AS n2, m.embedding_medcpt AS emb
            """
            params: dict[str, Any] = {
                "names": source_names,
                "edge_types": list(allowed_edges)
            }
        else:
            cypher = """
            MATCH (n)-[r]-(m)
            WHERE n.name IN $names
            RETURN labels(n) AS n1_labels, n.name AS n1, type(r) AS r1, 
                   labels(m) AS n2_labels, m.name AS n2, m.embedding_medcpt AS emb
            """
            params = {"names": source_names}

        try:
            # Rank neighbours by cosine similarity to the rewritten query
            async with self.driver.session() as session:        # type: ignore
                result = await session.run(cypher, **params)
                rows = await result.data()

            groups: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                sim = (
                    _cosine_sim(rewritten_query_vec, row["emb"])
                    if row.get("emb") else 0.0
                )
                # 'next' function: retrieve the first element
                n1_type = next((l for l in row["n1_labels"] if l != "KGNode"), "Entity")
                n2_type = next((l for l in row["n2_labels"] if l != "KGNode"), "Entity")
                
                groups[row["n1"]].append({
                    "n1_type": n1_type,
                    "n1": row["n1"],
                    "r1": row["r1"], 
                    "n2_type": n2_type,
                    "n2": row["n2"],
                    "sim": sim
                })

            final_triples: list[dict] = []
            for n1 in groups:
                ranked = sorted(groups[n1], key=lambda x: x["sim"], reverse=True)
                final_triples.extend(ranked[:top_n_per_node])
            return final_triples

        except Exception as e:
            logger.error(f"Neighbour search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Intent resolution
    # ------------------------------------------------------------------

    def _resolve_edge_types(self, intents: list[str]) -> set[str] | None:
        """Merge allowed edge types across multiple intents.
        Returns None if any intent is 'general' (allow all edges)."""
        merged: set[str] = set()
        for intent in intents:
            allowed = INTENT_EDGE_FILTER.get(intent, INTENT_EDGE_FILTER["general"])
            if allowed is None:
                return None
            merged.update(allowed)
        return merged if merged else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        entity_article_embeddings: list[list[float]],
        rewritten_query_vec: list[float],
        intents: list[str] = ["general"],
        top_k: int = 3,
        hop1_m: int = 10,
        hop2_n: int = 5,
        hop2_cap: int = 50
    ) -> list[dict]:
        """Full two-stage KG retrieval.

        Args:
            entity_article_embeddings:
                One Article-Encoder vector per extracted entity string.
                Used for Stage 1 anchor search (A-E vs A-E, same-space).
            rewritten_query_vec:
                Query-Encoder(rewritten_query) vector.
                Used for Stage 2 neighbour ranking (Q-E vs A-E, MedCPT cross-space).
            intents:
                Query intent list from query_extractor.py.
                Controls allowed edge types at each hop.
            top_k:    Max anchor nodes per entity in Stage 1.
            hop1_m:   Max 1-hop neighbours per anchor.
            hop2_n:   Max 2-hop neighbours per 1-hop node.
            hop2_cap: Hard cap on total 2-hop triples.

        Returns:
            List of linearized paths:
            [
               {"text": "Entity: A -[rel]-> Disease: B", "metadata": {"sim": 0.9, "hop": 1}},
               ...
            ]
        """
        if not self.driver:
            logger.warning("Neo4j driver unavailable — Skipping KG retrieval...")
            return []

        allowed_edges = self._resolve_edge_types(intents)

        # Stage 1: anchor search (same-space: A-E vs A-E)
        anchors = await self._find_anchors(entity_article_embeddings, top_k)
        if not anchors:
            return []

        # Stage 2a: 1-hop expansion (cross-space: Q-E rewritten_query vs A-E nodes)
        import asyncio
        hop1 = await self._get_neighbors(anchors, rewritten_query_vec, allowed_edges, hop1_m)
        logger.info(f"[Stage 2a]: 1-hop expansion → {len(hop1)} triples")

        # Stage 2b: 2-hop expansion (same cross-space ranking)
        hop1_nodes = list({t["n2"] for t in hop1})       # set comprehension
        hop2 = await self._get_neighbors(hop1_nodes, rewritten_query_vec, allowed_edges, hop2_n)

        if len(hop2) > hop2_cap:
            hop2 = hop2[:hop2_cap]
            logger.info(f"[Stage 2b]: Capped 2-hop at {hop2_cap} triples")
        else:
            logger.info(f"[Stage 2b]: 2-hop → {len(hop2)} triples")

        results = []
        for t in hop1:
            results.append({
                "text": f"{t['n1_type']}: {t['n1']} -[{t['r1']}]-> {t['n2_type']}: {t['n2']}",
                "metadata": {"sim": t["sim"], "hop": 1}
            })
        for t in hop2:
            results.append({
                "text": f"{t['n1_type']}: {t['n1']} -[{t['r1']}]-> {t['n2_type']}: {t['n2']}",
                "metadata": {"sim": t["sim"], "hop": 2}
            })

        return results