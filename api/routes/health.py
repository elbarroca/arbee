"""
Health check endpoints
"""
from datetime import datetime
from fastapi import APIRouter, Depends
from api.dependencies import get_copy_agent
from .agents.copy_trading_agent import CopyTradingAgent
from config.settings import settings

router = APIRouter()


@router.get("/")
async def root(copy_agent: CopyTradingAgent = Depends(get_copy_agent)):
    """Simple health check endpoint"""
    trader_count = len(copy_agent.copy_list) if hasattr(copy_agent, 'copy_list') else 0
    return {
        "status": "online",
        "service": "POLYSEER Copy Trading API",
        "timestamp": datetime.utcnow().isoformat(),
        "tracked_traders": trader_count,
        "dry_run_mode": settings.DRY_RUN_MODE
    }


@router.get("/health")
async def health(copy_agent: CopyTradingAgent = Depends(get_copy_agent)):
    """Detailed health check with system status"""
    active_traders = copy_agent.get_active_traders()
    trader_list = copy_agent.copy_list if hasattr(copy_agent, 'copy_list') else {}
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "copy_trading_enabled": settings.ENABLE_COPY_TRADING,
        "dry_run_mode": settings.DRY_RUN_MODE,
        "tracked_traders": len(trader_list),
        "active_traders": len(active_traders),
        "paused_traders": len([t for t in trader_list.values() if t.status == "paused"]),
        "webhook_providers": {
            "alchemy": bool(settings.ALCHEMY_API_KEY),
            "quicknode": bool(settings.QUICKNODE_API_KEY),
            "moralis": bool(settings.MORALIS_API_KEY)
        }
    }

