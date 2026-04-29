"""
Constructing the final context prompt to be fed into the LLM. It applies the
'Head-Tail replacement' strategy to optimize LLM's attention over the long
context (avoid lost-in-the-middle)
"""
from typing import Dict, List, Any, Optional, Tuple

def apply_head_tail_replacement(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorders a list of retrieved chunks/paths using Head-Tail placement."""
    if not items:
        return []
    
    n = len(items)
    reordered: List[Optional[Dict[str, Any]]] = [None] * n
    head_idx = 0
    tail_idx = n - 1

    for i, item in enumerate(items):
        if i % 2 == 0:
            # Place at head
            reordered[head_idx] = item
            head_idx += 1
        else:
            # Place at tail
            reordered[tail_idx] = item
            tail_idx -= 1

    return [x for x in reordered if x is not None]


def build_context_string(items: List[Dict[str, Any]]) -> str:
    """Construct the final context string from the reordered items."""
    context_parts = []

    for idx, item in enumerate(items, 1):
        item_type = item.get("type", "unknown")
        content = item.get("text", "")
        
        if item_type == "text_retrieval":
            context_parts.append(f"[Document {idx}]: {content}")
        elif item_type == "kg_retrieval":
            context_parts.append(f"[Knowledge Graph {idx}]: {content}")
        else:
            context_parts.append(f"[Context {idx}]: {content}")

    return "\n\n".join(context_parts)


def build_prompts(query: str, retrieved_items: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Build the decoupled System and User prompts for the LLM.
    
    Args: 
        query: The user's rewritten query
        retrieved_items: The reranked list of context items

    Returns:
        Tuple[str, str]: A tuple containing the (system_prompt, user_prompt)
    """
    # Apply head-tail-placement
    reordered_items = apply_head_tail_replacement(retrieved_items)

    # Build context string
    context_str = build_context_string(reordered_items)

    # Construct the final prompts
    system_prompt = (
        "You are a helpful and knowledgeable Medical AI Assistant. "
        "Your task is to answer the user's question accurately based ONLY on the provided context.\n\n"
        "Instructions:\n"
        "1. If the context does not contain enough information to answer the question, clearly state that you do not have enough information. Do not make up facts.\n"
        "2. Maintain a professional, objective, and empathetic medical tone.\n"
    )

    user_prompt = (
        f"Context:\n{context_str}\n\n"
        f"User question:\n{query}\n\n"
        f"Answer:"
    )

    return system_prompt, user_prompt

