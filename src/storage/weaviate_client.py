import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
import numpy as np
from typing import List, Dict
from config.logging_config import logger

CHILD_COLLECTION = "ChildChunks"

class WeaviateChildStore:
    """Weaviate client for storing and searching child chunks.
    Handles both vector search and BM25 keyword search.
    """
    def __init__(
            self, 
            url: str = "http://localhost:8080", 
            grpc_port: int = 50051
        ):
        # connect_to_local: use when database is local
        self.client = weaviate.connect_to_local(
            host=url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(url.split(":")[-1] if ":" in url.split("//")[-1] else 8080),
            grpc_port=grpc_port
        )
        logger.info(f"Connected to Weaviate {self.client.is_ready()}")

    def create_collection(self):
        """Create child chunks collections if not exists"""
        if self.client.collections.exists(CHILD_COLLECTION):
            logger.info(f"Collection '{CHILD_COLLECTION} already exists, skipping...'")
            return
        
        self.client.collections.create(
            name=CHILD_COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),   # We create our own vector embeddings
            properties=[
                Property(name="parent_id", data_type=DataType.TEXT),
                Property(name="pmid", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT)
            ]
        )
        logger.info(f"Created collection '{CHILD_COLLECTION}'!")

    def delete_collection(self):
        """Delete collection"""
        if self.client.collections.exists(CHILD_COLLECTION):
            self.client.collections.delete(CHILD_COLLECTION)
            logger.info(f"Deleted collection '{CHILD_COLLECTION}'")

    def insert_children(self, children: List[Dict], vectors: np.ndarray):
        """Batch insert child chunks with their MedCPT vectors"""
        collection = self.client.collections.get(CHILD_COLLECTION)

        with collection.batch.dynamic() as batch:   # Use batch insertion
            for i, child in enumerate(children):
                batch.add_object(
                    properties={
                        "parent_id": child["parent_id"],
                        "pmid": child["pmid"],
                        "text": child["text"]
                    },
                    vector=vectors[i].tolist()
                )
        
        logger.info(f"Inserted {len(children)} child chunks into Weaviate")

    def vector_search(self, query_vector: np.ndarray, limit: int = 20) -> List[Dict]:
        """Search child chunks by cosine similarity. Return lists of 
        {'parent_id', 'pmid', 'text', 'score'}"""
        collection = self.client.collections.get(CHILD_COLLECTION)

        response = collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )

        results = []
        for obj in response.objects:
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            results.append({
                "parent_id": obj.properties["parent_id"],
                "pmid": obj.properties["pmid"],
                "text": obj.properties["text"],
                "score": 1 - distance,    # cosine similarity
            })

        return results
    
    def bm25_search(self, query_text: str, limit: int = 20) -> List[Dict]:
        """Search child chunks by BM25 keyword matching. Return list of 
        {'parent_id', 'pmid', 'text', 'score'}"""
        collection = self.client.collections.get(CHILD_COLLECTION)

        response = collection.query.bm25(
            query=query_text,
            limit=limit,
            return_metadata=MetadataQuery(score=True)
        )

        results = []
        for obj in response.objects:
            results.append({
                "parent_id": obj.properties["parent_id"],
                "pmid": obj.properties["pmid"],
                "text": obj.properties["text"],
                "score": obj.metadata.score
            })

        return results
    
    def count(self) -> int:
        collection = self.client.collections.get(CHILD_COLLECTION)
        total_count = collection.aggregate.over_all(total_count=True).total_count
        return total_count if total_count is not None else 0

    def close(self):
        """Close the HTTP/gRPC connection to the server"""
        self.client.close()
    
    