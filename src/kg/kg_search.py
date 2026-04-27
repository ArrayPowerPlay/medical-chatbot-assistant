from config.logging_config import logger
from src.kg.neo4j_client import Neo4jClient
from src.kg.kg_linearization import PathLinearizer


class KGSearch:
    """Orchestrates KG retrieval: Stage 1 anchor search + Stage 2 2-hop
    traversal + rule-based linearization.
    """

    def __init__(self):
        self.client = Neo4jClient()

    def search(
        self,
        entity_article_embeddings: list[list[float]],
        rewritten_query_vec: list[float],
        intents: list[str] = ["general"],
        top_k: int = 3,
        hop1_m: int = 10,
        hop2_n: int = 5,
        hop2_cap: int = 50
    ) -> str:
        """Full KG retrieval pipeline: 2-stage retrieval → KG linearization.

        Args:
            entity_article_embeddings:
                [Article-Encoder(e) for e in entities].
                Stage 1 anchor search — same-space      (A-E vs A-E).
            rewritten_query_vec:
                Query-Encoder(rewritten_query) vector.
                Stage 2 neighbour ranking — cross-space (Q-E vs A-E).
            intents:
                Intent list from query_extractor.py.
                Controls edge-type filters at each hop.
            top_k:    Max anchor nodes per entity (Stage 1).
            hop1_m:   Max 1-hop neighbours per anchor.
            hop2_n:   Max 2-hop neighbours per 1-hop node.
            hop2_cap: Hard cap on total 2-hop triples.

        Returns:
            Newline-joined linearized triple strings, or "" if nothing found.
        """
        result = self.client.retrieve_subgraph(
            entity_article_embeddings=entity_article_embeddings,
            rewritten_query_vec=rewritten_query_vec,
            intents=intents,
            top_k=top_k,
            hop1_m=hop1_m,
            hop2_n=hop2_n,
            hop2_cap=hop2_cap
        )

        hop1_triples = result.get("hop1", [])
        hop2_triples = result.get("hop2", [])

        if not hop1_triples and not hop2_triples:
            logger.info("[KG Search] No triples returned")
            return ""

        paths = PathLinearizer.linearize(hop1_triples, hop2_triples)

        logger.info(
            f"[KG Search]: {len(hop1_triples)} 1-hop + {len(hop2_triples)} 2-hop "
            f"=> Generated {len(paths)} independent paths"
        )
        # Return newline-joined paths
        return "\n".join(paths)

    def cleanup(self) -> None:
        self.client.close()