"""
Trader management endpoints
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from .agents.copy_trading_agent import CopyTrader, CopyTradingAgent

from api.dependencies import get_copy_agent
from api.schemas.traders import (
    TraderResponse,
    TraderCreate,
    TraderListResponse
)
from api.exceptions import TraderNotFoundError, TraderAlreadyExistsError, TraderCriteriaNotMetError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/traders", tags=["traders"])


@router.get("", response_model=TraderListResponse)
async def list_traders(
    status: Optional[str] = Query(None, description="Filter by status: active, paused"),
    min_pnl_30d: Optional[float] = Query(None, description="Minimum 30-day PnL"),
    copy_agent: CopyTradingAgent = Depends(get_copy_agent)
):
    """List all tracked traders with optional filters"""
    traders = list(copy_agent.copy_list.values()) if hasattr(copy_agent, 'copy_list') else []
    
    # Apply filters
    if status:
        traders = [t for t in traders if t.status == status]
    
    if min_pnl_30d is not None:
        traders = [t for t in traders if t.pnl_30d >= min_pnl_30d]
    
    # Convert to response models
    trader_responses = [
        TraderResponse(
            wallet_address=t.wallet_address,
            trader_name=t.trader_name,
            pnl_30d=t.pnl_30d,
            pnl_90d=t.pnl_90d,
            pnl_all_time=t.pnl_all_time,
            win_rate=t.win_rate,
            trade_count=t.trade_count,
            avg_position_size=t.avg_position_size,
            sharpe_equivalent=t.sharpe_equivalent,
            categories_traded=t.categories_traded,
            wallet_age_days=t.wallet_age_days,
            added_date=t.added_date,
            status=t.status,
            last_trade_time=t.last_trade_time
        )
        for t in traders
    ]
    
    trader_list = copy_agent.copy_list if hasattr(copy_agent, 'copy_list') else {}
    active_count = len([t for t in trader_list.values() if t.status == "active"])
    paused_count = len([t for t in trader_list.values() if t.status == "paused"])
    
    return TraderListResponse(
        total=len(trader_list),
        active=active_count,
        paused=paused_count,
        traders=trader_responses
    )


@router.get("/{wallet_address}", response_model=TraderResponse)
async def get_trader(
    wallet_address: str,
    copy_agent: CopyTradingAgent = Depends(get_copy_agent)
):
    """Get trader details by wallet address"""
    trader_list = copy_agent.copy_list if hasattr(copy_agent, 'copy_list') else {}
    trader = trader_list.get(wallet_address.lower())
    
    if not trader:
        raise TraderNotFoundError(wallet_address)
    
    return TraderResponse(
        wallet_address=trader.wallet_address,
        trader_name=trader.trader_name,
        pnl_30d=trader.pnl_30d,
        pnl_90d=trader.pnl_90d,
        pnl_all_time=trader.pnl_all_time,
        win_rate=trader.win_rate,
        trade_count=trader.trade_count,
        avg_position_size=trader.avg_position_size,
        sharpe_equivalent=trader.sharpe_equivalent,
        categories_traded=trader.categories_traded,
        wallet_age_days=trader.wallet_age_days,
        added_date=trader.added_date,
        status=trader.status,
        last_trade_time=trader.last_trade_time
    )


@router.post("", response_model=TraderResponse)
async def add_trader(
    trader_data: TraderCreate,
    copy_agent: CopyTradingAgent = Depends(get_copy_agent)
):
    """Add a new trader to the copy list"""
    wallet_address = trader_data.wallet_address.lower()
    trader_list = copy_agent.copy_list if hasattr(copy_agent, 'copy_list') else {}
    
    # Check if trader already exists
    if wallet_address in trader_list:
        raise TraderAlreadyExistsError(wallet_address)
    
    # Create trader object
    trader = CopyTrader(
        wallet_address=wallet_address,
        trader_name=trader_data.trader_name,
        pnl_30d=trader_data.pnl_30d,
        pnl_90d=trader_data.pnl_90d,
        pnl_all_time=trader_data.pnl_all_time,
        win_rate=trader_data.win_rate,
        trade_count=trader_data.trade_count,
        avg_position_size=trader_data.avg_position_size,
        sharpe_equivalent=trader_data.sharpe_equivalent,
        categories_traded=trader_data.categories_traded,
        wallet_age_days=trader_data.wallet_age_days
    )
    
    # Try to add trader (will check criteria)
    success = copy_agent.add_trader(trader)
    
    if not success:
        raise TraderCriteriaNotMetError("Trader does not meet minimum criteria")
    
    logger.info(f"Added trader {wallet_address} to copy list")
    
    return TraderResponse(
        wallet_address=trader.wallet_address,
        trader_name=trader.trader_name,
        pnl_30d=trader.pnl_30d,
        pnl_90d=trader.pnl_90d,
        pnl_all_time=trader.pnl_all_time,
        win_rate=trader.win_rate,
        trade_count=trader.trade_count,
        avg_position_size=trader.avg_position_size,
        sharpe_equivalent=trader.sharpe_equivalent,
        categories_traded=trader.categories_traded,
        wallet_age_days=trader.wallet_age_days,
        added_date=trader.added_date,
        status=trader.status,
        last_trade_time=trader.last_trade_time
    )


@router.post("/{wallet_address}/pause")
async def pause_trader(
    wallet_address: str,
    copy_agent: CopyTradingAgent = Depends(get_copy_agent)
):
    """Pause copying a specific trader"""
    success = copy_agent.pause_trader(wallet_address.lower())
    
    if not success:
        raise TraderNotFoundError(wallet_address)
    
    logger.info(f"Paused trader {wallet_address}")
    return {"status": "success", "trader": wallet_address, "action": "paused"}


@router.post("/{wallet_address}/resume")
async def resume_trader(
    wallet_address: str,
    copy_agent: CopyTradingAgent = Depends(get_copy_agent)
):
    """Resume copying a specific trader"""
    success = copy_agent.resume_trader(wallet_address.lower())
    
    if not success:
        raise TraderNotFoundError(wallet_address)
    
    logger.info(f"Resumed trader {wallet_address}")
    return {"status": "success", "trader": wallet_address, "action": "resumed"}


@router.delete("/{wallet_address}")
async def remove_trader(
    wallet_address: str,
    copy_agent: CopyTradingAgent = Depends(get_copy_agent)
):
    """Remove a trader from the copy list"""
    success = copy_agent.remove_trader(wallet_address.lower())
    
    if not success:
        raise TraderNotFoundError(wallet_address)
    
    logger.info(f"Removed trader {wallet_address} from copy list")
    return {"status": "success", "trader": wallet_address, "action": "removed"}


