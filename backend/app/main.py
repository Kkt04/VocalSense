import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_audio import router as audio_router
from backend.app.api.v1.transcription import router as transcription_router
from backend.app.api.v1.emotion import router as emotion_router
from backend.app.api.v1.conversation import router as conversation_router
from backend.app.db.mongodb import mongodb_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vocalsense")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager handling MongoDB connections."""
    logger.info("Initializing MongoDB connection manager...")
    await mongodb_manager.connect()
    yield
    logger.info("Closing MongoDB connection manager...")
    await mongodb_manager.close()


def create_app() -> FastAPI:
    """Application factory for VocalSense FastAPI backend."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "VocalSense API — Backend service for the emotion-aware AI conversational companion. "
            "Handles speech input, audio validation, transcription, acoustic analysis, and conversation intelligence."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Correlation ID & Performance Logging Middleware
    @app.middleware("http")
    async def correlation_id_and_timing_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = req_id
        start_time = time.time()

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            "[%s] %s %s -> %d (%.1fms)",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"[{req_id}] Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            headers={"X-Request-ID": req_id},
            content={
                "success": False,
                "error": "An unexpected server error occurred.",
                "detail": str(exc) if settings.DEBUG else None,
                "code": "INTERNAL_SERVER_ERROR",
                "request_id": req_id,
            },
        )

    # Include Routers with /api and /api/v1 prefix
    app.include_router(health_router, prefix="/api")
    app.include_router(audio_router, prefix="/api")
    app.include_router(transcription_router, prefix="/api/v1")
    app.include_router(emotion_router, prefix="/api/v1")
    app.include_router(conversation_router, prefix="/api/v1")

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "online",
            "docs": "/docs",
            "health": "/api/health",
            "component_health": "/api/health/components",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
