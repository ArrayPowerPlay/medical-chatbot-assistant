"""
Chat endpointf for the chatbot API.

Endpoints:
    POST /api/chat         - Ask a medical question
"""
import json
import asyncio
from fastapi import Request, APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from config.logging_config import logger
from api.schemas.request import ChatRequest
from api.dependencies import get_current_user

router = APIRouter()


@router.post("/chat", summary="Ask a medical question", tags=["Chat"])
async def chat(request: ChatRequest, raw_request: Request, user: dict = Depends(get_current_user)):
    """
    Process a user question and return a streaming response through the RAG pipeline.
    """
    pipeline = raw_request.app.state.pipeline
    conv_store = raw_request.app.state.conv_store

    ### 1. Check if user = "guest" and if guest user exceeds number of question count
    if user.get("role") == "guest" and user.get("question_count", 0) >= 10:
        raise HTTPException(status_code=403, detail="Guest limit reached. Please register.")

    conversation_id = request.conversation_id
    if conversation_id and not conv_store.conversation_exists(conversation_id, user_id=user["id"]):
        logger.warning(f"[Chat] Conversation ID: {conversation_id} not found or unauthorized. Creating new...")
        conversation_id = None
    
    if conversation_id is None:
        title = request.question[:80] + ("..." if len(request.question) > 80 else "")
        conversation_id = conv_store.create_conversation(title=title, user_id=user["id"])

    ### 2. Load history
    history = conv_store.get_history(conversation_id, limit=10)

    # Save user message immediately to prevent it from disappearing if stream is aborted
    conv_store.add_message(conversation_id, "user", request.question)

    ### 3. Run streaming generation
    async def stream_generator():
        try:
            from src.pipeline.rag_pipeline import RunConfig
            run_config = RunConfig()
            
            # Yield conversation ID first for frontend redirection
            yield f"event: metadata\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
            
            full_answer = ""
            async for chunk in pipeline.run_stream(
                query=request.question,
                history=history,
                conversation_id=conversation_id,
                config=run_config
            ):
                if chunk.startswith("event: final_answer"):
                    # Extract final answer to save it to DB
                    data_str = chunk.split("data: ", 1)[1].strip()
                    data = json.loads(data_str)
                    full_answer = data.get("answer", "")
                else:
                    yield chunk
            
            # Save to db after successful generation
            msg_id = conv_store.add_message(conversation_id, "assistant", full_answer)
            yield f"event: message_id\ndata: {json.dumps({'message_id': msg_id})}\n\n"
            
            ### 5. Auto-set title for a conversation
            # (Moved to creation time for instant UI feedback)
            # if not request.conversation_id:
            #     title = request.question[:80] + ("..." if len(request.question) > 80 else "")
            #     conv_store.update_conversation(conversation_id, title=title)
                
            conv_store.increment_question_count(user["id"])
            
        except asyncio.CancelledError:
            logger.info("Client disconnected during stream.")
            raise
        except Exception as e:
            logger.error(f"[Chat] Stream error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'detail': 'Internal pipeline error.'})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")