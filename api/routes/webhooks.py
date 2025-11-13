"""
Webhook endpoints for receiving blockchain events
"""
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Header, HTTPException, Depends
from fastapi.responses import JSONResponse

from api.dependencies import get_trade_processor, get_copy_agent, get_trade_executor
from api.schemas.webhooks import WebhookProcessingResponse
from betting.copy_trading import TradeSignalProcessor
from agents.copy_trading_agent import CopyTradingAgent
from clients.trade.trade_executor import TradeExecutor
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


async def process_webhook_event(
    event_data: Dict[str, Any],
    provider: str,
    trade_processor: TradeSignalProcessor,
    copy_agent: CopyTradingAgent,
    trade_executor: TradeExecutor
) -> Dict[str, Any]:
    """
    Process a webhook event and execute copy trading logic.

    Args:
        event_data: Raw webhook payload
        provider: Provider name ("alchemy", "quicknode", "moralis")
        trade_processor: Trade signal processor
        copy_agent: Copy trading agent
        trade_executor: Trade executor for executing trades

    Returns:
        Processing result dict
    """
    if not settings.ENABLE_COPY_TRADING:
        logger.info("Copy trading disabled, ignoring webhook event")
        return {"action": "skipped", "reason": "copy_trading_disabled"}

    # Parse webhook based on provider
    if provider == "alchemy":
        from clients.web3.alchemy import AlchemyWebhooksClient
        client = AlchemyWebhooksClient()
        parsed_event = client.parse_webhook_event(event_data)
    else:
        logger.warning(f"Unknown provider: {provider}")
        return {"action": "error", "reason": "unknown_provider"}

    if not parsed_event:
        logger.debug(f"No trade signal extracted from {provider} webhook")
        return {"action": "ignored", "reason": "not_trade_signal"}

    logger.info(f"Parsed trade signal from {provider}: {parsed_event.get('wallet_address', 'unknown')}")

    # Validate and process trade signal
    validated_signal = await trade_processor.process_webhook_event(parsed_event)

    if not validated_signal:
        logger.info(f"Trade signal filtered out (validation failed)")
        return {"action": "filtered", "reason": "validation_failed"}

    # Execute trade using TradeExecutor (handles DRY_RUN mode internally)
    logger.info(f"Executing copy trade for {validated_signal.wallet_address} on market {validated_signal.market_slug}")
    
    try:
        executed_trade = await trade_executor.execute_trade(validated_signal)
        
        return {
            "action": "executed" if executed_trade.status == "filled" else "rejected",
            "trader": validated_signal.wallet_address.lower(),
            "market": validated_signal.market_slug,
            "side": validated_signal.side,
            "size_usd": validated_signal.size_usd,
            "status": executed_trade.status,
            "fill_price": executed_trade.fill_price,
            "slippage_bps": executed_trade.slippage_bps,
            "error_message": executed_trade.error_message
        }
    except Exception as e:
        logger.error(f"Trade execution failed: {e}", exc_info=True)
        return {
            "action": "error",
            "reason": str(e),
            "trader": validated_signal.wallet_address.lower(),
            "market": validated_signal.market_slug
        }


@router.post("/alchemy", response_model=WebhookProcessingResponse)
async def alchemy_webhook(
    request: Request,
    trade_processor: TradeSignalProcessor = Depends(get_trade_processor),
    copy_agent: CopyTradingAgent = Depends(get_copy_agent),
    trade_executor: TradeExecutor = Depends(get_trade_executor)
):
    """
    Receive and process Alchemy webhook events.

    Alchemy sends address activity and contract events here.
    """
    try:
        event_data = await request.json()
        logger.info(f"Received Alchemy webhook: {event_data.get('type', 'unknown')}")

        # Process the webhook event
        result = await process_webhook_event(event_data, "alchemy", trade_processor, copy_agent, trade_executor)

        return JSONResponse(
            status_code=200,
            content={"status": "success", **result}
        )
    except Exception as e:
        logger.error(f"Error processing Alchemy webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quicknode", response_model=WebhookProcessingResponse)
async def quicknode_webhook(
    request: Request,
    trade_processor: TradeSignalProcessor = Depends(get_trade_processor),
    copy_agent: CopyTradingAgent = Depends(get_copy_agent),
    trade_executor: TradeExecutor = Depends(get_trade_executor)
):
    """
    Receive and process QuickNode webhook events.

    QuickNode sends filtered blockchain events here.
    """
    try:
        event_data = await request.json()
        logger.info(f"Received QuickNode webhook: {len(event_data.get('events', []))} events")

        # Process the webhook event
        result = await process_webhook_event(event_data, "quicknode", trade_processor, copy_agent, trade_executor)

        return JSONResponse(
            status_code=200,
            content={"status": "success", **result}
        )
    except Exception as e:
        logger.error(f"Error processing QuickNode webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/moralis", response_model=WebhookProcessingResponse)
async def moralis_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None),
    trade_processor: TradeSignalProcessor = Depends(get_trade_processor),
    copy_agent: CopyTradingAgent = Depends(get_copy_agent),
    trade_executor: TradeExecutor = Depends(get_trade_executor)
):
    """
    Receive and process Moralis Stream webhook events.

    Moralis sends stream events with signature verification.
    """
    try:
        event_data = await request.json()
        logger.info(f"Received Moralis webhook: {event_data.get('tag', 'unknown')} stream")

        # Verify signature if x_signature is provided
        if x_signature:
            logger.debug(f"Moralis signature: {x_signature[:20]}...")

        # Process the webhook event
        result = await process_webhook_event(event_data, "moralis", trade_processor, copy_agent, trade_executor)

        return JSONResponse(
            status_code=200,
            content={"status": "success", **result}
        )
    except Exception as e:
        logger.error(f"Error processing Moralis webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

