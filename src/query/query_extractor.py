import json
from groq import Groq
from config.settings import settings
from config.logging_config import logger
from typing import Dict, List
from src.kg.schema import QueryIntent


class QueryExtractor:
    """
    Use LLM to extract key medical entities for querying the knowledge graph (Neo4j)
    Centralized entities: Disease, Symptom, Drug
    """
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL
        self._valid_intents = {qi.value for qi in QueryIntent}

    def extract(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from the rewritten query
        Returns: A dictionary containing lists of diseases, symptoms, drugs and intents
        """
        system_prompt = (
            """You are an expert clinical query extractor. Given a medical query, perform two tasks simultaneously and return a single JSON object.

            ### TASK 1:
            Extract medical entities into exactly three categories: "diseases", "symptoms", "drugs".

            ### TASK 2:
            Classify the query into exactly ONE of these intent categories below (use ["general"] only when the query is truly ambiguous)
            - "symptom_lookup"          : asking about symptoms or signs of a disease
            - "treatment_lookup"        : asking what drugs or therapies treat a disease
            - "mechanism_lookup"        : asking how a drug works, its mechanism of action
            - "side_effect_lookup"      : asking about side effects or adverse effects of a drug
            - "contraindication_lookup" : asking when a drug should NOT be used
            - "disease_relation"        : asking about related diseases, comorbidities, subtypes
            - "genetic_association"     : asking about genes or proteins linked to a disease
            - "drug_target_lookup"      : asking what proteins or targets a drug acts on
            - "general"                 : query is ambiguous or spans multiple intents

            ### RULES:
            1. JSON FORMAT ONLY: You must return a completely valid, parseable JSON object. Do not wrap it in markdown code blocks (e.g., no ```json). Do not include any introductory or concluding text.
            2. EMPTY STATES: If no entity belongs to a category, output an empty array [].

            ### EXPECTED OUTPUT SCHEMA:
            {
            "diseases": ["name1", "name2"],
            "symptoms": ["name1", "name2"],
            "drugs": ["name1"],
            "intents": ["treatment_lookup"]
            }
        """
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system", "content": system_prompt,
                    "role": "user",   "content": query
                }],
                temperature=0.0,
                response_format={"type": "json_object"}   # JSON type casting
            )
            response_content = completion.choices[0].message.content
            if response_content is None:
                # Handle the case where there is no content from the API
                logger.info("Error in QueryExtractor: API returned no content.")
                return self._empty_result()

            entities = json.loads(response_content)
            raw_intents = entities.get("intents", ["general"])
            intents = [i for i in raw_intents if i in self._valid_intents] or ['general']

            return {
                "diseases": entities.get("diseases", []),
                "drugs": entities.get("drugs", []),
                "symptoms": entities.get("symptoms", []),
                "intents": intents
            }                    
        except Exception as e:
            logger.info(f"Error in QueryExtractor {e}")
            return self._empty_result()    

    @staticmethod
    def _empty_result() -> dict:    # static method does not contain 'self'
        return {"diseases": [], "symptoms": [], "drugs": [], "intent": ["general"]}