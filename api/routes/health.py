"""
Health check endpoint for the chatbot API.
"""
from fastapi import APIRouter
from config.logging_config import logger
from api.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, 
            summary="Service health check", tags=["System"]) # 'tags': group API endpoints in Swagger UI
def health_check() -> HealthResponse:
    services: dict = {}

    ### Weaviate
    try:
        import weaviate
        from config.settings import settings
        client = weaviate.connect_to_local(
            host=settings.WEAVIATE_URL.replace("http://", "").split(":")[0],
            port=int(settings.WEAVIATE_URL.split(":")[-1]),
            grpc_port=settings.WEAVIATE_GRPC_PORT
        )
        services["weaviate"] = "healthy" if client.is_ready() else "unhealthy"
        client.close()
    except Exception as e:
        logger.warning(f"[Health Check]: Weaviate health check failed: {e}")
        services["weaviate"] = "unhealthy"

    ### Neo4j
    try:
        from neo4j import GraphDatabase
        from config.settings import settings
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD) if settings.NEO4J_PASSWORD else None
        )
        driver.verify_connectivity()
        services["neo4j"] = "healthy"
        driver.close()
    except Exception as e:
        logger.warning(f"[Health Check]: Neo4j health check failed: {e}")
        services["neo4j"] = "unhealthy"

    ### PostgreSQL
    try:
        import psycopg2
        from config.settings import settings
        conn = psycopg2.connect(
            host=settings.POSTGRE_HOST,
            port=settings.POSTGRE_PORT,
            user=settings.POSTGRE_USER,
            password=settings.POSTGRE_PASSWORD,
            dbname=settings.POSTGRE_DB
        )
        conn.close()
        services["postgre"] = "healthy"
    except Exception as e:
        logger.warning(f"[Health Check]: PostgreSQL health check failed: {e}")
        services["postgre"] = "unhealthy"
    
    overall = "healthy" if all(v == "healthy" for v in services.values()) else "unhealthy"
    return HealthResponse(status=overall, services=services)

