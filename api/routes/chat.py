"""
Chat endpointf for the chatbot API.

Endpoints:
    POST /api/chat         - Ask a medical question
"""
from fastapi import Request, APIRouter, HTTPException
from config.logging_config import logger
from api.schemas.request import ChatRequest
from api.schemas.response import ChatResponse, SourceItem

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Ask a medical question", tags=["Chat"])
def chat(request: ChatRequest, raw_request: Request) -> ChatResponse:
    """
    Process a user question and return a valid response through the RAG pipeline.
    
    Args:
        request: Validated ChatRequest body.
        raw_request: FastAPI request carries app-level state.
    
    Returns:
        ChatResponse: Generated answer, source citations, and conversation_id.
    """
    pipeline = raw_request.app.state.pipeline
    conv_store = raw_request.app.state.conv_store

    ### 1. Resolve conversation ID
    conversation_id = request.conversation_id
    if conversation_id and not conv_store.conversation_exists(conversation_id):
        logger.warning(f"[Chat] Conversation ID: {conversation_id} not found. Creating new...")
        conversation_id = None
    
    if conversation_id is None:
        conversation_id = conv_store.create_conversation()

    ### 2. Load history
    history = conv_store.get_history(conversation_id, limit=10)

    ### 3. Run pipeline
    try:
        use_citations = request.use_citations if request.use_citations is not None else settings.USE_CITATIONS
        result = pipeline.run(
            query=request.question,
            history=history,
            conversation_id=conversation_id,
            use_citations=use_citations,
        )
    except Exception as e:
        logger.error(f"[Chat]: Pipeline error: {e}", exc_info=True)   # logger adds the full stack trace of the exception to the log
        raise HTTPException(status_code=500, detail="Internal pipeline error.")
    
    ### 4. Add new messages into conversation with 'conversation_id' id
    answer = result["answer"]
    conv_store.add_message(conversation_id, "user", request.question) # user's raw question
    conv_store.add_message(conversation_id, "assistant", answer)

    ### 5. Auto-set title for a conversation
    if not request.conversation_id:
        title = request.question[:80] + ("..." if len(request.question) > 80 else "")
        conv_store.update_title(conversation_id, title)
    
    ### 6. Build the response
    sources = [
        SourceItem(
            source_type=src.get("source_type", "unknown"),
            content=src.get("text", src.get("content", "")),
            pmid=src.get("pmid"),
        )
        for src in result.get("sources", [])
    ]

    return ChatResponse(answer=answer, sources=sources, conversation_id=conversation_id)