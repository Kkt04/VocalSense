import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_audio import router as audio_router
from backend.app.api.v1.transcription import router as transcription_router
from backend.app.api.v1.emotion import router as emotion_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vocalsense")


def create_app() -> FastAPI:
    """Application factory for VocalSense FastAPI backend."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "VocalSense API — Backend service for the emotion-aware AI conversational companion. "
            "Handles speech input, audio validation, transcription, and acoustic analysis."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "An unexpected server error occurred.",
                "detail": str(exc) if settings.DEBUG else None,
                "code": "INTERNAL_SERVER_ERROR",
            },
        )

    # Include Routers with /api and /api/v1 prefix
    app.include_router(health_router, prefix="/api")
    app.include_router(audio_router, prefix="/api")
    app.include_router(transcription_router, prefix="/api/v1")
    app.include_router(emotion_router, prefix="/api/v1")

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "online",
            "docs": "/docs",
            "health": "/api/health",
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
