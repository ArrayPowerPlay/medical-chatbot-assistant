"""
FastAPI application entry point for the chatbot.

Responsibilities:
    - Configure CORS for frontend access.
    - Initialize shared resources (RAG pipeline, conversation store) at startup.
    - Mount API routes.
"""
from contextlib import asynccontextmanager    # Initialize/clean up resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles   # Allows access to static files via URL

from config.settings import settings
from config.logging_config import setup_logging, logger
from api.routes import chat, health, conversations

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle.
    
    Startup: logging, RAG Pipeline, conversation store.
    Shutdown: close conversation store connection."""
    setup_logging()
    logger.info(f"[Startup]: Initializing RAG pipeline...")

    from src.pipeline.rag_pipeline import RAGPipeline
    from src.storage.conversation_store import ConversationStore

    app.state.pipeline = RAGPipeline()
    app.state.conv_store = ConversationStore()

    logger.info("[Startup]: All service ready.")

    yield   # Application running...

    logger.info("[Shutdown]: Releasing resources...")
    app.state.conv_store.close()
    logger.info("[Shutdown]: Cleanup complete.")


app = FastAPI(
    title="MedRAG-KG chatbot API",
    description="Medical QA chatbot powered by KG-augmented RAG",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # Allow the browser to send authentication information with the request
    allow_methods=["*"],
    allow_headers=["*"]       # Accept all request headers
)

app.include_router(chat.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(health.router, prefix="/api")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")