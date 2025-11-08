"""
Trade execution endpoints
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends

from api.dependencies import get_trade_processor, get_trade_executor
from api.schemas.trades import TradeExecutionResponse, TradeSignalRequest
from api.exceptions import TradeExecutionError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/trades", tags=["trades"])


@router.post("/execute", response_model=TradeExecutionResponse)
async def execute_trade(
    signal: TradeSignalRequest,
    trade_processor=Depends(get_trade_processor),
    trade_executor=Depends(get_trade_executor)
):
    """
    Execute a trade manually (for testing/debugging).
    
    Note: In production, trades are executed automatically via webhooks.
    """
    try:
        # Validate signal
        validated_signal = await trade_processor.process_webhook_event({
            "wallet_address": signal.wallet_address,
            "market_slug": signal.market_slug,
            "token_address": signal.token_address,
            "side": signal.side,
            "amount": signal.size_usd,
            "price": signal.price,
            "transaction_hash": signal.transaction_hash,
            "block_number": signal.block_number,
            "provider": signal.source_provider,
            "raw_event": signal.raw_event
        })
        
        if not validated_signal:
            raise TradeExecutionError("Signal validation failed")
        
        # Execute trade using TradeExecutor (handles DRY_RUN mode internally)
        executed_trade = await trade_executor.execute_trade(validated_signal)
        
        if executed_trade.status == "rejected":
            raise TradeExecutionError(executed_trade.error_message or "Trade execution rejected")
        
        return TradeExecutionResponse(
            status=executed_trade.status,
            market_slug=executed_trade.market_slug,
            side=executed_trade.side,
            size_usd=executed_trade.size_usd,
            fill_price=executed_trade.fill_price or executed_trade.expected_price,
            executed_at=executed_trade.timestamp
        )
            
    except TradeExecutionError:
        raise
    except Exception as e:
        logger.error(f"Trade execution failed: {e}", exc_info=True)
        raise TradeExecutionError(str(e))

