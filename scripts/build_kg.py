"""
Import filtered PrimeKG into Neo4j and store MedCPT node embeddings.

Pipeline:
    - Create Neo4j uniqueness constraints and B-tree indexes on node_id.
    - Batch import nodes using UNWIND + MERGE per node label.
    - Batch import relationships using UNWIND + MERGE per relationship type,
    grouping by (rel_type, x_type, y_type) for dynamic schema support.
    - Build enriched text for every node: "{Label}: {name}"
    - Encode with MedCPT-Article-Encoder 
    - Write embedding_medcpt back into each Neo4j node using the common :KGNode label 
    - Create vector index 'medcpt_embeddings' on the :KGNode label
"""

import sys
import pandas as pd
from pathlib import Path
import numpy as np
from typing import cast, LiteralString
from neo4j import GraphDatabase, Driver, Query

project_root = Path(__file__).resolve().parent.parent
if str(project_root) in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from config.logging_config import logger, setup_logging
from src.kg.schema import NODE_TYPE_TO_LABEL
from src.embeddings.medcpt_embedder import MedCPTEmbedder

NODES_PATH = Path("data/knowledge_graph/nodes_filtered.csv")
RELATION_PATH = Path("data/knowledge_graph/relationships_filtered.csv")

NODE_BATCH_SIZE     = 500       # Nodes per Neo4j UNWIND transaction
RELATION_BATCH_SIZE = 1_000     # Relationships per Neo4j UNWIND transaction
EMBED_BATCH_SIZE    = 256       # Texts per MedCPT forward pass


def _run_in_batches(driver: Driver, cypher: str, data: list[dict], batch_size: int, desc: str = "") -> None:
    """
    Execute a parametrized Cypher statement on data split into fixed-size batches

    Args:
        driver: An active Neo4j driver instance.
        cypher: A Cypher statement that uses the parameter $batch via UNWIND
        data: The full list of parameter dictionaries to process
        batch_size: Expected maximum number of dicts per transaction
        desc: Description for logging process

    Returns: 
        None
    """
    total_batches = (len(data) + batch_size - 1) // batch_size
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        # Open a temporary session in Neo4j driver
        with driver.session() as session:
            session.run(cypher, batch=batch)      # type: ignore

        # Simple progress logging
        current_batch = (i // batch_size) + 1
        if current_batch % max(1, (total_batches // 10)) == 0 or current_batch == total_batches:
            logger.info(f" Progress: {current_batch}/{total_batches} batches processed {desc}")


def create_constraints(driver: Driver) -> None:
    """
    Create uniqueness constraints on node_id for every mapped node label.
    Optimized for importing batch filtered data 
    A uniqueness constraint implicitly creates a B-tree index.
    """
    with driver.session() as session:
        for label in NODE_TYPE_TO_LABEL.values():
            constraint_name = f"{label.lower()}_node_id"
            # Create label and index for node_id of that label
            cypher_text = f"""  
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS    
                FOR (n:{label}) REQUIRE n.node_id IS UNIQUE
                """
            cypher = Query(cast(LiteralString, cypher_text))
            session.run(cypher)
            logger.info(f"Constraint ensured: {label}.node_id IS UNIQUE")


def import_nodes(driver: Driver, nodes_df: pd.DataFrame) -> None:
    """
    Batch-import nodes into Neo4j using UNWIND + MERGE, grouped by label

    Each node receives two labels:
        - Its domain label: (Disease, Drug, GeneProtein, EffectPhenotype)
        - :KGNode (secondary label, used for vector search in Stage 1 retrieval)

    MERGE is idempotent: re-running the script will update existing nodes rather 
    than creating duplicates
    """
    for primekg_type, label in NODE_TYPE_TO_LABEL.items():
        subset = nodes_df[nodes_df["type"] == primekg_type]
        if subset.empty:
            logger.warning(f"No nodes found for type '{primekg_type}' (Label: {label})")
            continue

        data = subset[["node_id", 'name', "source"]].to_dict("records")  # Convert DataFrame to list of dicts

        cypher = f"""
        UNWIND $batch AS row
        MERGE (n:{label}:KGNode {{node_id: row.node_id}})
        SET n.name = row.name,
            n.source = row.source
        """
        logger.info(f"Importing {len(data):,} nodes for {label}...")
        _run_in_batches(driver, cypher, data, NODE_BATCH_SIZE, desc=f"({label} nodes)")


def import_relationships(driver: Driver, relations_df: pd.DataFrame) -> None:
    """
    Batch-import relationships into Neo4j dynamically grouping by (rel_type, x_type, y_type)
    """
    grouped = relations_df.groupby(["rel_type", "x_type", "y_type"])

    for (rel_type, x_type, y_type), group in grouped:
        x_label = NODE_TYPE_TO_LABEL.get(x_type)    # type: ignore
        y_label = NODE_TYPE_TO_LABEL.get(y_type)    # type: ignore

        # Encounter unknown types
        if not x_label or not y_label:
            logger.warning(f"Skipping relation {rel_type} because of unknown types: {x_type} -> {y_type}")
            continue

        data = group[["x_id", "y_id"]].to_dict("records")   # Convert DataFrame to list of dicts

        cypher = f"""
        UNWIND $batch as row
        MATCH (x:{x_label}:KGNode {{node_id: row.x_id}})
        MATCH (y:{y_label}:KGNode {{node_id: row.y_id}})
        MERGE (x)-[:{rel_type}]->(y)
        """

        logger.info(f"Importing {len(data):,} '{rel_type}' edges ({x_label})->({y_label})...")
        _run_in_batches(driver, cypher, data, RELATION_BATCH_SIZE, desc=f"({rel_type} edges)")


def embed_and_store(driver: Driver, nodes_df: pd.DataFrame) -> None:
    """
    Generate MedCPT-Article-Encoder embeddings for all nodes and write them to Neo4j

    Enriched text for each node embedding: "{Label}: {name}"
    """
    embedder = MedCPTEmbedder(mode='article')

    nodes_df = nodes_df.copy()
    nodes_df["label"] = nodes_df["type"].map(NODE_TYPE_TO_LABEL)
    valid_nodes = nodes_df.dropna(subset=["label"]).copy()
    valid_nodes["enriched_text"] = valid_nodes["label"] + ": " + valid_nodes["name"]

    texts = valid_nodes["enriched_text"].tolist()
    node_ids = valid_nodes["node_id"].tolist()

    logger.info(f"Encoding {len(texts):,} nodes with MedCPT-Article-Encoder...")

    # Generate embeddings 
    embeddings: np.ndarray = embedder.embed_texts(texts, batch_size=EMBED_BATCH_SIZE)
    
    logger.info("Writing embedding to Neo4j nodes using :KGNode label...")

    data = [
        {"node_id": nid, "embedding": emb.tolist()}
        for nid, emb in zip(node_ids, embeddings)
    ]

    cypher = """
    UNWIND $batch AS row
    MATCH (n:KGNode {node_id: row.node_id})
    SET n.embedding_medcpt = row.embedding
    """

    _run_in_batches(driver, cypher, data, NODE_BATCH_SIZE, desc="(embeddings)")
    logger.info("Embeddings successfully stored")

    embedder.close()


def create_vector_index(driver: Driver) -> None:
    """Create the 'medcpt_node_embeddings' vector index on the :KGNode label.
    Uses cosine similarity
    """
    with driver.session() as session:
        session.run("""
            CREATE VECTOR INDEX medcpt_node_embeddings IF NOT EXISTS
            FOR (n:KGNode)
            ON (n.embedding_medcpt)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}
        """)
    logger.info("Create vector index 'medcpt_node_embeddings' successfully!")


def main() -> None:
    """Run the full Neo4j import and embedding pipeline"""
    logger.info("Pipeline building Neo4j KG started")

    if not NODES_PATH.exists() or not RELATION_PATH.exists():
        raise FileNotFoundError(
            f"Filtered CSVs not found in data/knowledge_graph/."
        )

    logger.info("Loading filtered CSV data into memory...")
    nodes_df = pd.read_csv(NODES_PATH, dtype=str)    # Force all read columns to become string type
    rels_df = pd.read_csv(RELATION_PATH, dtype=str)
    logger.info(f"Loaded {len(nodes_df):,} nodes and {len(rels_df):,} edges")

    driver = GraphDatabase.driver(
        settings.NEO4J_URL,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

    try:
        logger.info("---Step 1: Enforcing constraints---")
        create_constraints(driver)

        logger.info("---Step 2: Importing nodes---")
        import_nodes(driver, nodes_df)

        logger.info("---Step 3: Importing relationships---")
        import_relationships(driver, rels_df)

        logger.info("---Step 4: Generating and storing embeddings---")
        embed_and_store(driver, nodes_df)

        logger.info("---Step 5: Creating vector index---")
        create_vector_index(driver)

    finally:
        driver.close()

    logger.info("Pipeline completed!")


if __name__ == "__main__":
    setup_logging()
    main()