"""
Construct the final prompts fed into the generator model.

Type-conditional prompting: the system prompt is tailored based on the
BioASQ question type (summary | list | yesno | factoid) provided by
QueryAnalyzer (classified by the LLM).
If question_type is missing or unrecognised, defaults to 'summary'.
"""

from typing import Any, Dict, List, Optional, Tuple


# Type-specific prompt instructions
_TYPE_INSTRUCTIONS: Dict[str, str] = {
    "summary": (
        "Provide a comprehensive answer in 3–5 complete sentences covering all key aspects. "
        "Use paragraph form. Do not use bullet points or numbered lists."
    ),
    "list": (
        "Enumerate all relevant items found in the context. "
        "Begin with one brief introductory sentence, then name each item explicitly. "
        "Include every distinct member mentioned in the context."
    ),
    "yesno": (
        "Begin your answer with 'Yes' or 'No', then provide a 1–2 sentence explanation "
        "that justifies the answer using evidence from the context."
    ),
    "factoid": (
        "Provide a direct, concise answer in 1 sentence. "
        "Name the specific entity, value, or location asked for. "
        "Do not add unnecessary background."
    ),
}


# ---------------------------------------------------------------------------
# Head-tail reordering
# ---------------------------------------------------------------------------

def apply_head_tail_replacement(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorder retrieved items using head-tail placement."""
    if not items:
        return []

    reordered: List[Optional[Dict[str, Any]]] = [None] * len(items)
    head_idx = 0
    tail_idx = len(items) - 1

    for i, item in enumerate(items):
        if i % 2 == 0:
            reordered[head_idx] = item
            head_idx += 1
        else:
            reordered[tail_idx] = item
            tail_idx -= 1

    return [item for item in reordered if item is not None]


def build_context_string(items: List[Dict[str, Any]], use_citations: bool = True) -> str:
    """Construct the final context string from retrieved items."""
    context_parts = []

    for idx, item in enumerate(items, 1):
        item_type = item.get("source_type", "unknown")
        content = item.get("text", item.get("content", ""))
        pmid = item.get("pmid", "")

        if item_type == "text_retrieval":
            if use_citations and pmid:
                context_parts.append(f"[Document {idx} (PMID: {pmid})]: {content}")
            else:
                context_parts.append(f"[Document {idx}]: {content}")
        elif item_type == "kg_retrieval":
            context_parts.append(f"[Knowledge Graph {idx}]: {content}")
        else:
            context_parts.append(f"[Context {idx}]: {content}")

    return "\n\n".join(context_parts)


def _collect_text_pmids(items: List[Dict[str, Any]]) -> List[str]:
    """Return an ordered, deduplicated list of PMIDs from text-retrieval items.

    Used to build the PMID checklist appended to the user prompt so the model
    has an explicit reminder of every citable source in the context.
    """
    seen: set = set()
    pmids: List[str] = []
    for item in items:
        if item.get("source_type") == "text_retrieval":
            pmid = item.get("pmid", "").strip()
            if pmid and pmid not in seen:
                seen.add(pmid)
                pmids.append(pmid)
    return pmids


def build_prompts(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    use_head_tail_placement: bool = True,
    use_citations: bool = True,
    question_type: Optional[str] = None,
) -> Tuple[str, str]:
    """Build system and user prompts for the generator.

    Args:
        query: The (rewritten) user question.
        retrieved_items: Ranked list of retrieved passages/KG paths.
        use_head_tail_placement: Whether to apply head-tail context reordering.
        use_citations: Whether to instruct the LLM to cite PMIDs.
        question_type: One of 'summary' | 'list' | 'yesno' | 'factoid',
            provided by QueryAnalyzer. Defaults to 'summary' if missing.
    """
    items_for_prompt = (
        apply_head_tail_replacement(retrieved_items)
        if use_head_tail_placement
        else retrieved_items
    )
    context_str = build_context_string(items_for_prompt, use_citations=use_citations)

    resolved_type = question_type if question_type in _TYPE_INSTRUCTIONS else "summary"
    type_instruction = _TYPE_INSTRUCTIONS[resolved_type]

    system_prompt = (
        "You are a helpful and knowledgeable Medical AI Assistant. "
        "Your task is to answer the user's question accurately based ONLY on the provided context.\n\n"
        "Instructions:\n"
        "1. If the context does not contain enough information to answer the question, clearly state "
        "that you do not have enough information. Do not make up facts.\n"
        "2. Maintain a professional, objective, and empathetic medical tone.\n"
        f"3. {type_instruction}\n"
        "4. Start your answer directly with the information asked. "
        "Do not open with phrases like 'Based on the provided context' or 'According to the documents'.\n"
    )

    if use_citations:
        system_prompt += (
            "5. Citation rules — follow strictly:\n"
            "   a. After EVERY factual claim, cite ALL PMIDs from the context that support it, "
            "using brackets: [12345] for one source, [12345, 67890] for multiple.\n"
            "   b. Do NOT invent or guess PMIDs. Only cite PMIDs that appear in the provided "
            "context documents (labeled as 'PMID: XXXXX').\n"
        )

    # Build the PMID checklist that appears at the end of the user prompt.
    # Providing an explicit list gives the model a concrete set to verify
    # against, directly addressing the rank-7-to-12 omission pattern.
    pmid_checklist = ""
    if use_citations:
        text_pmids = _collect_text_pmids(items_for_prompt)
        if text_pmids:
            pmid_checklist = (
                "\n\nAvailable PMIDs (all text documents in the context above — "
                "cite every one whose content you used):\n"
                + ", ".join(text_pmids)
            )

    user_prompt = (
        f"Context:\n{context_str}\n\n"
        f"User question:\n{query}"
        f"{pmid_checklist}\n\n"
        f"Answer:"
    )

    return system_prompt, user_prompt
