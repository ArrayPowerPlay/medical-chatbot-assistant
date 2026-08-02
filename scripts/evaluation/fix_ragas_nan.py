import json
import math
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.logging_config import setup_logging, logger
from config.settings import settings
from scripts.evaluation.shared import generation_bioasq_common as common
from scripts.evaluation.shared.config_helper import load_and_apply_config

from src.pipeline.rag_pipeline import RAGPipeline
from src.query.query_analyzer import QueryAnalyzer
from src.embeddings.medcpt_embedder import MedCPTEmbedder
from src.storage.weaviate_client import AsyncWeaviateChildStore
from src.storage.parent_store import ParentStore
from src.kg.neo4j_client import Neo4jClient
from src.generation.llm_generator import LLMGenerator
from src.reranking.rrf import RRFManager
from src.reranking.cross_encoder import CrossEncoderReranker
from src.generation.kg_merger import KGPathMerger


async def fix_checkpoint_for_method(method_name: str, use_kg: bool, use_vector: bool, use_bm25: bool):
    output_dir = project_root / "results" / "test_results" / "bioasq" / method_name / "generation"
    checkpoint_path = output_dir / "ragas_checkpoint.json"
    predictions_path = output_dir / "predictions.jsonl"
    
    if not checkpoint_path.exists() or not predictions_path.exists():
        logger.warning(f"Missing files for {method_name}. Skipping.")
        return
        
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
        
    nan_ids = []
    for qid, metrics in checkpoint.items():
        has_nan = False
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                has_nan = True
                break
        if has_nan:
            nan_ids.append(qid)
            
    if not nan_ids:
        logger.info(f"No NaN values found in {method_name}.")
        return
        
    logger.info(f"Found {len(nan_ids)} questions with NaN metrics in {method_name}. Preparing for re-evaluation.")
    
    # Load predictions
    predictions = {}
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            predictions[rec["question_id"]] = rec
            
    ragas_evaluator = common.initialize_ragas_evaluator(enabled=True)
    if ragas_evaluator is None:
        logger.error("Failed to initialize RAGAS.")
        return
    
    logger.info("Initializing RAG pipeline for retrieval...")
    pipeline = RAGPipeline(
        query_analyzer=QueryAnalyzer(),
        query_embedder=MedCPTEmbedder(mode='query'),
        entity_embedder=MedCPTEmbedder(mode='article'),
        search_engine=AsyncWeaviateChildStore(),
        parent_store=ParentStore(settings.SQLITE_PARENT_DB_PATH),
        kg_searcher=Neo4jClient(),
        rrf_manager=RRFManager(),
        cross_encoder_reranker=CrossEncoderReranker(),
        kg_merger=KGPathMerger(),
        llm_generator=LLMGenerator() # We won't use it but need it to init
    )
    
    ragas_inputs = []
    
    try:
        for i, qid in enumerate(nan_ids):
            pred = predictions.get(qid)
            if not pred:
                logger.error(f"Prediction for {qid} not found.")
                continue
                
            query = pred.get("body", "")
            rewritten_query = pred.get("rewritten_query", query)
            generated_answer = pred.get("generated_answer", "")
            ideal_answer = pred.get("ideal_answer", [])
            
            logger.info(f"[{i+1}/{len(nan_ids)}] Retrieving context for {qid}...")
            
            query_vectors = await pipeline.query_embedder.embed_texts(rewritten_query)
            query_vector = query_vectors[0].tolist()
            
            vector_results, bm25_results, kg_results = await pipeline.parallel_retriever.retrieve(
                query_text=rewritten_query,
                query_vector=query_vector,
                entity_article_embeddings=[],
                intents=["general"],
                vector_top_k=settings.VECTOR_TOP_K,
                keyword_top_k=settings.KEYWORD_TOP_K,
                child_fetch_limit=settings.CHILD_FETCH_LIMIT,
                kg_top_k=settings.KG_TOP_K,
                kg_hop1_m=settings.KG_HOP1_M,
                kg_hop2_n=settings.KG_HOP2_N,
                kg_hop2_cap=settings.KG_HOP2_CAP,
                original_query_text=query,
            )
            
            if not use_vector: vector_results = []
            if not use_bm25: bm25_results = []
            if not use_kg: kg_results = []
            
            rrf_results = pipeline.rrf_manager.rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results
            )
            
            ranked_text, ranked_kg = await pipeline.cross_encoder_reranker.rerank(
                query=rewritten_query,
                rrf_results=rrf_results,
                kg_results=kg_results,
                top_m=settings.VECTOR_TOP_K,
                top_n=settings.RERANK_KG_TOP_N,
            )
            
            sources = ranked_text + ranked_kg
            contexts = [
                text.strip()
                for item in sources
                for text in [item.get("text", item.get("content", ""))]
                if isinstance(text, str) and text.strip()
            ]
            
            # Save retrieved contexts back into pred
            pred["contexts"] = contexts
            
            ragas_inputs.append({
                "question_index": 0,
                "question_id": qid,
                "question": query,
                "answer": generated_answer,
                "contexts": contexts,
                "references": ideal_answer,
            })
            
        logger.info(f"Re-evaluating RAGAS for {len(ragas_inputs)} questions...")
        
        # Use a temporary checkpoint so run_ragas_chunked_with_checkpoint doesn't skip them
        temp_checkpoint = output_dir / "ragas_nan_fix_checkpoint.json"
        
        max_retries = 5
        for attempt in range(max_retries):
            # Check which inputs are still NaN in the MAIN checkpoint
            still_nan_inputs = []
            for inp in ragas_inputs:
                qid = inp["question_id"]
                metrics = checkpoint.get(qid, {})
                has_nan = False
                for k, v in metrics.items():
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        has_nan = True
                        break
                # Also if it's completely missing, we need to run it
                if not metrics or has_nan:
                    still_nan_inputs.append(inp)
                    
            if not still_nan_inputs:
                logger.info("All NaNs have been successfully resolved!")
                break
                
            logger.info(f"Attempt {attempt+1}/{max_retries}: Running RAGAS for {len(still_nan_inputs)} questions with NaNs...")
            if temp_checkpoint.exists():
                temp_checkpoint.unlink()
                
            scores_by_qid = common.run_ragas_chunked_with_checkpoint(
                ragas_inputs=still_nan_inputs,
                ragas_evaluator=ragas_evaluator,
                checkpoint_path=temp_checkpoint,
                chunk_size=5,
            )
            
            # Merge back only valid non-NaN scores
            for qid, scores in scores_by_qid.items():
                has_nan = any((isinstance(v, float) and math.isnan(v)) for v in scores.values())
                if not has_nan:
                    checkpoint[qid] = scores
                    
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            
        remaining_nans = sum(1 for _, m in checkpoint.items() if any(isinstance(v, float) and math.isnan(v) for v in m.values()))
        logger.info(f"Fixed records. Remaining NaNs in {method_name}: {remaining_nans}")
        
        # We should also update predictions.jsonl with the contexts we found
        with open(predictions_path, "w", encoding="utf-8") as f:
            for pred in predictions.values():
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        logger.info(f"Updated {predictions_path} with new contexts.")
        
    finally:
        await common._close_pipeline(pipeline)


async def main():
    setup_logging()
    load_and_apply_config("generation")
    
    logger.info("=== Fixing Baseline Vector ===")
    await fix_checkpoint_for_method(
        method_name="baseline_vector",
        use_kg=False,
        use_vector=True,
        use_bm25=False
    )
    
    logger.info("=== Fixing No KG Hybrid ===")
    await fix_checkpoint_for_method(
        method_name="no_kg_hybrid",
        use_kg=False,
        use_vector=True,
        use_bm25=True
    )


if __name__ == "__main__":
    asyncio.run(main())
