"""
Pydantic schemas for trade endpoints
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TradeSignalRequest(BaseModel):
    """Trade signal request from webhook"""
    wallet_address: str
    market_slug: str
    token_address: Optional[str] = None
    side: str = Field(..., description="BUY or SELL")
    size_usd: float
    price: float
    transaction_hash: str
    block_number: Optional[int] = None
    source_provider: str = "unknown"
    raw_event: Dict[str, Any] = Field(default_factory=dict)


class TradeExecutionResponse(BaseModel):
    """Trade execution response"""
    trade_id: Optional[str] = None
    status: str = Field(..., description="executed, rejected, pending")
    market_slug: str
    side: str
    size_usd: float
    fill_price: Optional[float] = None
    slippage_bps: Optional[float] = None
    error_message: Optional[str] = None
    executed_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class PaperTradeResponse(BaseModel):
    """Paper trading log entry response"""
    id: int
    recorded_at: datetime
    market_slug: str
    token_id: str
    side: str
    size_usd: float
    expected_price: Optional[float] = None
    fill_price: Optional[float] = None
    slippage_bps: Optional[float] = None
    realized_pnl: float = 0.0
    position_qty_after: Optional[float] = None
    position_cost_basis: Optional[float] = None
    signal_source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TradeStatsResponse(BaseModel):
    """Trading statistics response"""
    total_trades: int
    executed_trades: int
    rejected_trades: int
    total_volume_usd: float
    avg_trade_size_usd: float
    avg_slippage_bps: float
    win_rate: float
    total_pnl: float

