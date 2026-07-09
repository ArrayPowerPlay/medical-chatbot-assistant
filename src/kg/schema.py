"""
KG Schemas - Node types, Relationship mappings, and Intent-based edge filters

Define the subset of PrimeKG that we use in MedRAG-KG:
    1. NodeType / KEEP_NODE_TYPES       — which PrimeKG node types to keep
    2. NODE_TYPE_TO_LABEL               — PrimeKG type string → Neo4j label
    3. RELATION_TO_LABEL                — PrimeKG display_relation → Neo4j rel type
    4. NEO4J_INDEXES                    — standard property indexes to create in Neo4j
    5. QueryIntent / INTENT_EDGE_FILTER — intent categories from query_extractor.py
                                         and allowed edge types per hop for each intent

Focus: Drug ←→ Disease ←→ Target (Gene/Protein)
"""

from enum import Enum


class NodeType(Enum):
    """PrimeKG node types for MedRAG-KG"""
    DISEASE          = "disease"
    DRUG             = "drug"
    GENE_PROTEIN     = "gene/protein"
    EFFECT_PHENOTYPE = "effect/phenotype"


# Set of node types to keep when filtering PrimeKG
KEEP_NODE_TYPES: set[str] = {nt.value for nt in NodeType}

# Mapping PrimeKG x_type/y_type -> Neo4j node label
NODE_TYPE_TO_LABEL: dict[str, str] = {
    "disease":          "Disease",
    "drug":             "Drug",
    "gene/protein":     "GeneProtein",
    "effect/phenotype": "EffectPhenotype"   # covers symptoms AND side effects
}

# PrimeKG display_relation → Neo4j relationship type
RELATION_TO_LABEL: dict[str, str] = {
    # Drug → Disease
    "indication":       "TREATS",
    "contraindication": "CONTRAINDICATES",
    "off-label use":    "OFF_LABEL_USE",

    # Drug → GeneProtein 
    "target":           "TARGETS",
    "enzyme":           "METABOLIZED_BY",
    "transporter":      "TRANSPORTED_BY",
    "carrier":          "CARRIED_BY",

    # Drug → EffectPhenotype
    "side effect":      "HAS_SIDE_EFFECT",

    # Disease → EffectPhenotype
    "phenotype present": "PRESENTS",
    "phenotype absent":  "PHENOTYPE_ABSENT",

    # GeneProtein → Disease
    "associated with":  "ASSOCIATED_WITH",

    # Disease → Disease (hierarchy)
    "parent-child":     "SUBTYPE_OF",
}

# Standard property indexes (B-tree) for fast Cypher lookup
NEO4J_INDEXES: list[dict] = [
    {"label": "Disease",         "property": "name"},
    {"label": "Disease",         "property": "node_id"},
    {"label": "Drug",            "property": "name"},
    {"label": "Drug",            "property": "node_id"},
    {"label": "GeneProtein",     "property": "name"},
    {"label": "GeneProtein",     "property": "node_id"},
    {"label": "EffectPhenotype", "property": "name"},
    {"label": "EffectPhenotype", "property": "node_id"},
]


class QueryIntent(Enum):
    """
    All intent categories that query_extractor.py can produce.
    Multiple intents can be active simultaneously for one query.
    """
    SYMPTOM_LOOKUP          = "symptom_lookup"          # "What are the symptoms of X?"
    TREATMENT_LOOKUP        = "treatment_lookup"        # "What drugs treat X?"
    MECHANISM_LOOKUP        = "mechanism_lookup"        # "How does drug X work?"
    SIDE_EFFECT_LOOKUP      = "side_effect_lookup"      # "Side effects of drug X?"
    CONTRAINDICATION_LOOKUP = "contraindication_lookup" # "When is X contraindicated?"
    DISEASE_RELATION        = "disease_relation"        # "What diseases are related to X?"
    GENETIC_ASSOCIATION     = "genetic_association"     # "What genes are linked to X?"
    DRUG_TARGET_LOOKUP      = "drug_target_lookup"      # "What proteins does drug X target?"
    GENERAL                 = "general"                 # Ambiguous / multi-intent
    NO_RAG_NEEDED           = "no_rag_needed"           # Small talk or context-only answering


# Allowed Neo4j relationship types per intent.
# For specific intents, restrict traversal to semantically relevant edges only
INTENT_EDGE_FILTER: dict[str, set[str] | None] = {
    "symptom_lookup": {
        "PRESENTS",
        "PHENOTYPE_ABSENT",   # symptoms that DO NOT appear — useful for differential diagnosis
        "SUBTYPE_OF",
        "ASSOCIATED_WITH",
    },
    "treatment_lookup": {
        "TREATS",
        "OFF_LABEL_USE",
        "CONTRAINDICATES",
        "TARGETS",
        "HAS_SIDE_EFFECT",
    },
    "mechanism_lookup": {
        "TARGETS",
        "METABOLIZED_BY",
        "TRANSPORTED_BY",
        "CARRIED_BY",
        "ASSOCIATED_WITH",
    },
    "side_effect_lookup": {
        "HAS_SIDE_EFFECT",
        "CONTRAINDICATES",
        "PRESENTS",
    },
    "contraindication_lookup": {
        "CONTRAINDICATES",
        "OFF_LABEL_USE",
        "TREATS",
        "HAS_SIDE_EFFECT",
    },
    "disease_relation": {
        "SUBTYPE_OF",
        "ASSOCIATED_WITH",
        "PRESENTS",
        "PHENOTYPE_ABSENT",   # distinguishes related diseases by absent symptoms
    },
    "genetic_association": {
        "ASSOCIATED_WITH",
        "TARGETS",
    },
    "drug_target_lookup": {
        "TARGETS",
        "METABOLIZED_BY",
        "TRANSPORTED_BY",
        "CARRIED_BY",
        "ASSOCIATED_WITH",
    },
    "general": None,    # No restriction
    "no_rag_needed": None,
}