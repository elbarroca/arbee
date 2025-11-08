"""
Market data endpoints
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Query

from api.dependencies import get_polymarket_client
from api.exceptions import MarketNotFoundError
from .api_clients.polymarket import PolymarketClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/markets", tags=["markets"])


@router.get("/{market_slug}")
async def get_market(
    market_slug: str,
    polymarket_client: PolymarketClient = Depends(get_polymarket_client)
):
    """Get market details by slug"""
    try:
        market = await polymarket_client.gamma.get_market(market_slug)
        if not market:
            raise MarketNotFoundError(market_slug)
        return market
    except MarketNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get market {market_slug}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{market_slug}/orderbook")
async def get_orderbook(
    market_slug: str,
    depth: int = Query(10, ge=1, le=50, description="Orderbook depth"),
    polymarket_client: PolymarketClient = Depends(get_polymarket_client)
):
    """Get market orderbook"""
    try:
        market = await polymarket_client.gamma.get_market(market_slug)
        if not market:
            raise MarketNotFoundError(market_slug)
        
        token_ids = market.get("clobTokenIds", [])
        if not token_ids:
            raise MarketNotFoundError(market_slug)
        
        # Get orderbook for YES token (typically index 1)
        yes_token_id = token_ids[1] if len(token_ids) > 1 else token_ids[0]
        orderbook = polymarket_client.clob.get_orderbook(yes_token_id, depth=depth)
        
        return orderbook
    except MarketNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get orderbook for {market_slug}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

