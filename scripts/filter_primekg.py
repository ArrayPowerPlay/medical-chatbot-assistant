"""
Filter PrimeKG dataset down to the Drug-Disease-Target subset

Pipeline:
    1. Read kg.csv in chunks to avoid OOM error
    2. Keep only edges where:
        - Both endpoint node types are keep in KEEP_NODE_TYPES
        - display_relation is in RELATION_TO_LABEL
    3. Collect unique nodes from both sides of each kept edge
    4. Write two clean output files:
        - data/knowledge_graph/nodes_filtered.csv
        - data/knowledge_graph/relationships_filtered.csv
"""

import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.kg.schema import KEEP_NODE_TYPES, NODE_TYPE_TO_LABEL, RELATION_TO_LABEL
from config.logging_config import logger, setup_logging

RAW_KG_PATH = Path("data/knowledge_graph/kg.csv")
NODES_OUPUT_PATH = Path("data/knowledge_graph/nodes_filtered.csv")
RELATION_OUTPUT_PATH = Path("data/knowledge_graph/relationships_filtered.csv")

CHUNK_SIZE = 100_000   # Process in chunks

def _extract_nodes_from_chunk(chunk: pd.DataFrame) -> pd.DataFrame: 
    """Extract unique node rows from both endpoints (x side and y side) of an edge chunk.
    
    Args:
        Chunk: A filtered edge DataFrame containing columns x_id, x_type, x_name, 
        x_source, y_id, y_type, y_name, y_source

    Returns:
        A DataFrame with columns [node_id, type, name, source] containing nodes from both 
        sides of every edge in chunk.
    """
    rename_x = {"x_id": "node_id", "x_type": "type", "x_name": "name", "x_source": "source"}
    rename_y = {"y_id": "node_id", "y_type": "type", "y_name": "name", "y_source": "source"}

    x_nodes = chunk[list(rename_x)].rename(columns=rename_x)
    y_nodes = chunk[list(rename_y)].rename(columns=rename_y)

    return pd.concat([x_nodes, y_nodes], ignore_index=True)


def filter_edges(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply node-type and relation-type masks to one chunk of the raw kg.csv.
    
    Args:
        Chunk: One raw chunk from kg.csv containing all PrimeKG columns.
        
    Returns:
        A filtered sub-DataFrame keeping only rows where both endpoint
        types are in KEEP_NODE_TYPES and display_relation is in
        RELATION_TO_LABEL. Adds a 'rel_type' column for Neo4j label.
        Returns an empty DataFrame if no rows pass the filters.
        """    
    type_mask = (
        chunk["x_type"].isin(KEEP_NODE_TYPES) &
        chunk["y_type"].isin(KEEP_NODE_TYPES)
    )
    rel_mask = chunk["display_relation"].isin(RELATION_TO_LABEL)

    filtered = chunk[type_mask & rel_mask].copy()
    if not filtered.empty:
        # Map PrimeKG display_relation to the Neo4j relationship type label
        filtered["rel_type"] = filtered["display_relation"].map(RELATION_TO_LABEL)

    return filtered


def filter_primekg() -> None:
    """Read kg.csv in chunks, filter edges and nodes, then write CSVs.
    Accumulates filtered edges and unique nodes into lists of DataFrames,
    then performs a single pd.concat at the end.

    Returns:
        None. Writes nodes_filtered.csv and relationships_filtered.csv to disk.
    """
    logger.info(f"Reading {RAW_KG_PATH} in chunks of {CHUNK_SIZE:,} rows...")

    edge_chunks: list[pd.DataFrame] = []
    node_chunks: list[pd.DataFrame] = []

    total_rows = 0
    kept_rows = 0

    reader = pd.read_csv(
        RAW_KG_PATH,
        chunksize=CHUNK_SIZE,
        dtype=str            # Force all read columns to become string type
    )

    for chunk_idx, chunk in enumerate(reader):
        total_rows += len(chunk)
        filtered = filter_edges(chunk)
        kept_rows += len(filtered)

        if filtered.empty:
            continue

        edge_chunks.append(filtered[[
            "x_id", "x_type",
            "y_id", "y_type",
            "display_relation",
            "rel_type"
        ]])
        node_chunks.append(_extract_nodes_from_chunk(filtered))

        if (chunk_idx + 1) % 1000 == 0:
            logger.info(
                f"""
                Chunk {chunk_idx + 1}: {total_rows:,} rows scanned, {kept_rows} kept. 
                """
            )

    logger.info("Merging and deduplicating...")

    nodes_df = (
        pd.concat(node_chunks, ignore_index=True)
        .drop_duplicates(subset=["node_id"])
        .assign(label=lambda df: df["type"].map(NODE_TYPE_TO_LABEL))   # Create 'label' column for Neo4j node type mapping
        .reset_index(drop=True)
    )

    rels_df = (
        pd.concat(edge_chunks, ignore_index=True)
        .drop_duplicates(subset=["x_id", "y_id", "rel_type"])
        .reset_index(drop=True)
    )

    NODES_OUPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(NODES_OUPUT_PATH, index=False)    # Do not save index
    rels_df.to_csv(RELATION_OUTPUT_PATH, index=False)

    logger.info(f"Nodes: {len(nodes_df):,} nodes")
    logger.info(f"Edges: {len(rels_df):,} edges")
    logger.info(
        f"""
        Retention: {kept_rows:,}/{total_rows:,} rows ~ {kept_rows/total_rows*100:.2f}%
        """
    )

if __name__ == "__main__":
    setup_logging()
    filter_primekg()
