"""
This module calls an LLM to execute these works:
- Rewrite the query for optimal retrieval
- Extract medical entities and classify the user intent
This optimizes reduce latency and halves the number of LLM calls
"""
import json
from typing import Dict, List, Optional, Any
from groq import Groq

from config.settings import settings
from config.logging_config import logger
from src.kg.schema import QueryIntent


class QueryAnalyzer:
    """Analyzes user queries to rewrite them and extract structured entities and intents
    in a single LLM call."""
    def __init__(self):
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing in settings or environment variables.")
        
        self.client = Groq(api_key=api_key)
        self.model_name = settings.LLM_MODEL
        self.temperature = 0.0
        self._valid_intents = {qi.value for qi in QueryIntent}   # Set comprehension

    def analyze(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Analyzes the given query to perform rewriting, entity extraction, and intent classification.

        Returns:
            Dict[str, Any]: A dictionary containing the rewritten query, extracted entities, and intents.
        """
        system_prompt = (
            "You are an expert medical AI assistant for a RAG system. "
            "Your task is to analyze the user's raw conversational input and perform three tasks simultaneously.\n\n"
            "### TASK 1: QUERY REWRITING\n"
            "Transform the raw user input into a precise, standalone medical query optimized for retrieval.\n"
            "- Replace pronouns with exact clinical entities from conversation history (if provided).\n"
            "- Standardize brand names/layperson terms to clinical equivalents.\n"
            "- Strip conversational filler.\n"
            "- Fix typographical errors.\n\n"
            "### TASK 2: ENTITY EXTRACTION\n"
            "Extract medical entities from the REWRITTEN query into exactly four categories:\n"
            "- \"diseases\": disease or disorder names (e.g. \"Type 2 Diabetes\", \"Hypertension\")\n"
            "- \"effect_phenotypes\": symptoms, clinical findings, OR drug side effects "
            "(e.g. \"fever\", \"nausea\", \"chest pain\")\n"
            "- \"drugs\": drug, medication, or compound names (e.g. \"Metformin\", \"Aspirin\")\n"
            "- \"gene_proteins\": gene or protein names (e.g. \"BRCA1\", \"TP53\", \"insulin receptor\")\n"
            "If no entity belongs to a category, use an empty array [].\n\n"
            "### TASK 3: INTENT CLASSIFICATION\n"
            "Classify the REWRITTEN query into exactly one or many of these intent categories:\n"
            "- symptom_lookup, treatment_lookup, mechanism_lookup, side_effect_lookup, "
            "contraindication_lookup, disease_relation, genetic_association, drug_target_lookup, general\n\n"
            "### RULES:\n"
            "1. JSON FORMAT ONLY: Return a valid JSON object. Do not wrap in markdown or add explanations.\n"
            "2. EXACT OUTPUT SCHEMA REQUIRED:\n"
            "{\n"
            "  \"rewritten_query\": \"The optimized query string\",\n"
            "  \"diseases\": [\"name1\"],\n"
            "  \"effect_phenotypes\": [\"name1\"],\n"
            "  \"drugs\": [],\n"
            "  \"gene_proteins\": [],\n"
            "  \"intents\": [\"treatment_lookup\"]\n"
            "}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": f"Analyze this query: {query}"})

        try: 
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,                        # type: ignore
                temperature=self.temperature,
                response_format={"type": "json_object"}   # JSON type casting
            )

            response_content = completion.choices[0].message.content
            if not response_content:
                logger.error("[Query Analyzer]: API returned empty content.")
                return self._fallback_result(query)
            
            data = json.loads(response_content)

            raw_intents = data.get("intents", ["general"])
            valid_intents = [i for i in raw_intents if i in self._valid_intents]
            if not valid_intents:
                valid_intents = ["general"]

            return {
                "rewritten_query": data.get("rewritten_query", query),
                "diseases": data.get("diseases", []),
                "effect_phenotypes": data.get("effect_phenotypes", []),
                "drugs": data.get("drugs", []),
                "gene_proteins": data.get("gene_proteins", []),
                "intents": valid_intents
            }
        
        except Exception as e:
            logger.error(f"Error in query analyzer: {e}")
            return self._fallback_result(query)

    def _fallback_result(self, query: str) -> Dict[str, Any]:
        """Provides a safe fallback dictionary if the LLM call fails."""
        return {
            "rewritten_query": query,
            "diseases": [],
            "effect_phenotypes": [],
            "drugs": [],
            "gene_proteins": [],
            "intents": ["general"]
        }


