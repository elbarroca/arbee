"""
Pydantic schemas for trader endpoints
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class TraderResponse(BaseModel):
    """Trader response model"""
    wallet_address: str
    trader_name: Optional[str] = None
    pnl_30d: float = 0.0
    pnl_90d: float = 0.0
    pnl_all_time: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    avg_position_size: float = 0.0
    sharpe_equivalent: float = 0.0
    categories_traded: List[str] = Field(default_factory=list)
    wallet_age_days: int = 0
    added_date: datetime
    status: str = "active"
    last_trade_time: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TraderCreate(BaseModel):
    """Create trader request model"""
    wallet_address: str = Field(..., description="Wallet address")
    trader_name: Optional[str] = None
    pnl_30d: float = 0.0
    pnl_90d: float = 0.0
    pnl_all_time: float = 0.0
    win_rate: float = 0.5
    trade_count: int = 0
    avg_position_size: float = 0.0
    sharpe_equivalent: float = 0.0
    categories_traded: List[str] = Field(default_factory=list)
    wallet_age_days: int = 0


class TraderUpdate(BaseModel):
    """Update trader request model"""
    trader_name: Optional[str] = None
    pnl_30d: Optional[float] = None
    pnl_90d: Optional[float] = None
    pnl_all_time: Optional[float] = None
    win_rate: Optional[float] = None
    trade_count: Optional[int] = None
    avg_position_size: Optional[float] = None
    sharpe_equivalent: Optional[float] = None
    categories_traded: Optional[List[str]] = None
    wallet_age_days: Optional[int] = None


class TraderListResponse(BaseModel):
    """List traders response model"""
    total: int
    active: int
    paused: int
    traders: List[TraderResponse]


class TraderMetricsResponse(BaseModel):
    """Trader metrics response model"""
    wallet_address: str
    pnl_30d: float
    pnl_90d: float
    win_rate: float
    trade_count: int
    sharpe_equivalent: float
    avg_position_size: float
    last_trade_time: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

