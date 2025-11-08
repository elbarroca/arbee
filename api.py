"""
POLYSEER Copy Trading API Entry Point
"""
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from api.dependencies import initialize_dependencies, cleanup_dependencies
from api.routes import health, webhooks, traders, trades, markets, discovery
from api.exceptions import APIException
from config.settings import settings

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    logger.info("Starting POLYSEER Copy Trading API...")
    initialize_dependencies()
    yield
    # Shutdown
    logger.info("Shutting down POLYSEER Copy Trading API...")
    cleanup_dependencies()


# Create FastAPI app
app = FastAPI(
    title="POLYSEER Copy Trading API",
    description="API for copy trading system - trader discovery, trade execution, and market data",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code or "API_ERROR",
                "message": exc.detail
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred"
            }
        }
    )


# Include routers
app.include_router(health.router)
app.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])
app.include_router(traders.router)
app.include_router(trades.router)
app.include_router(markets.router)
app.include_router(discovery.router)


@app.get("/api/v1")
async def api_info():
    """API information endpoint"""
    return {
        "name": "POLYSEER Copy Trading API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "webhooks": "/webhooks/{provider}",
            "traders": "/api/v1/traders",
            "trades": "/api/v1/trades",
            "markets": "/api/v1/markets",
            "discovery": "/api/v1/discovery"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )

