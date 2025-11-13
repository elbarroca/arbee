"""
Pydantic schemas for unified provider responses

Standardized schemas for market data across all providers (Polymarket, Kalshi, etc.)
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class OrderbookLevel(BaseModel):
    """Single price level in orderbook"""
    price: float = Field(..., description="Price (0-1 for probability)")
    size: float = Field(..., description="Size in contracts or USD")


class UnifiedOrderbook(BaseModel):
    """Unified orderbook schema across all providers"""
    provider: str
    market_id: str
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    bids: List[OrderbookLevel] = Field(default_factory=list, description="Sorted descending by price")
    asks: List[OrderbookLevel] = Field(default_factory=list, description="Sorted ascending by price")
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float = Field(..., description="Absolute spread")
    spread_bps: int = Field(..., description="Spread in basis points")
    bid_liquidity: float = Field(..., description="Sum of top 10 bid levels (USD)")
    ask_liquidity: float = Field(..., description="Sum of top 10 ask levels (USD)")
    total_liquidity: float = Field(..., description="Min(bid_liquidity, ask_liquidity)")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "polymarket",
                "market_id": "0x123abc",
                "timestamp": "2025-01-15T12:00:00Z",
                "bids": [{"price": 0.52, "size": 100.0}],
                "asks": [{"price": 0.54, "size": 120.0}],
                "best_bid": 0.52,
                "best_ask": 0.54,
                "mid_price": 0.53,
                "spread": 0.02,
                "spread_bps": 200,
                "bid_liquidity": 5200.0,
                "ask_liquidity": 6480.0,
                "total_liquidity": 5200.0
            }
        }


class UnifiedMarket(BaseModel):
    """Unified market schema across all providers"""
    provider: str
    market_id: str
    slug: str
    question: str
    event_id: Optional[str] = None
    category: str = "general"
    outcomes: List[str] = Field(default_factory=lambda: ["YES", "NO"])
    yes_price: float = Field(..., description="YES outcome price (0-1)")
    no_price: float = Field(..., description="NO outcome price (0-1)")
    mid_price: float = Field(..., description="Mid-market price")
    volume: float = Field(default=0.0, description="Total volume in USD")
    liquidity: float = Field(default=0.0, description="Available liquidity in USD")
    spread: float = Field(default=0.0, description="Bid-ask spread")
    spread_bps: int = Field(default=0, description="Spread in basis points")
    end_date: Optional[str] = Field(None, description="Market close date (ISO 8601)")
    status: Literal["active", "closed", "settled"] = "active"
    last_updated: str = Field(..., description="Last update timestamp (ISO 8601)")
    orderbook: Optional[UnifiedOrderbook] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific data")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "polymarket",
                "market_id": "0x123abc",
                "slug": "will-btc-reach-100k",
                "question": "Will Bitcoin reach $100k by end of 2025?",
                "event_id": "evt_crypto_2025",
                "category": "crypto",
                "outcomes": ["YES", "NO"],
                "yes_price": 0.65,
                "no_price": 0.35,
                "mid_price": 0.65,
                "volume": 125000.0,
                "liquidity": 15000.0,
                "spread": 0.02,
                "spread_bps": 200,
                "end_date": "2025-12-31T23:59:59Z",
                "status": "active",
                "last_updated": "2025-01-15T12:00:00Z",
                "metadata": {}
            }
        }


class UnifiedEvent(BaseModel):
    """Unified event schema across all providers"""
    provider: str
    event_id: str
    slug: str
    title: str
    description: str = ""
    category: str = "general"
    start_date: Optional[str] = Field(None, description="Event start date (ISO 8601)")
    end_date: Optional[str] = Field(None, description="Event end date (ISO 8601)")
    total_volume: float = Field(default=0.0, description="Total volume across all markets (USD)")
    market_count: int = Field(default=0, description="Number of markets in this event")
    status: Literal["active", "closed", "settled"] = "active"
    markets: List[UnifiedMarket] = Field(default_factory=list, description="Markets in this event")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific data")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "polymarket",
                "event_id": "evt_crypto_2025",
                "slug": "crypto-predictions-2025",
                "title": "Crypto Predictions 2025",
                "description": "Will major cryptocurrencies hit key milestones in 2025?",
                "category": "crypto",
                "start_date": "2025-01-01T00:00:00Z",
                "end_date": "2025-12-31T23:59:59Z",
                "total_volume": 500000.0,
                "market_count": 5,
                "status": "active",
                "markets": [],
                "metadata": {}
            }
        }


class ProviderHealth(BaseModel):
    """Provider health check response"""
    provider: str
    status: Literal["healthy", "degraded", "unhealthy"]
    latency_ms: float
    last_check: str = Field(..., description="Last check timestamp (ISO 8601)")
    error: Optional[str] = None
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate (0-1)")
    rate_limit_remaining: Optional[int] = Field(None, description="Remaining API calls this period")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "polymarket",
                "status": "healthy",
                "latency_ms": 245.5,
                "last_check": "2025-01-15T12:00:00Z",
                "error": None,
                "cache_hit_rate": 0.72,
                "rate_limit_remaining": 850
            }
        }


class MultiProviderHealth(BaseModel):
    """Health status for all providers"""
    timestamp: str = Field(..., description="Health check timestamp (ISO 8601)")
    providers: List[ProviderHealth]
    overall_status: Literal["healthy", "degraded", "unhealthy"]

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-15T12:00:00Z",
                "providers": [
                    {
                        "provider": "polymarket",
                        "status": "healthy",
                        "latency_ms": 245.5,
                        "last_check": "2025-01-15T12:00:00Z",
                        "error": None
                    }
                ],
                "overall_status": "healthy"
            }
        }


class MarketPrice(BaseModel):
    """Simple market price response"""
    provider: str
    market_id: str
    price: float = Field(..., description="Market price (0-1 probability)")
    side: Literal["mid", "bid", "ask", "last"] = "mid"
    timestamp: str = Field(..., description="Price timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "polymarket",
                "market_id": "0x123abc",
                "price": 0.65,
                "side": "mid",
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }


class ProviderStats(BaseModel):
    """Provider statistics and metrics"""
    provider: str
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    avg_latency_ms: float
    errors_count: int
    rate_limit_hits: int
    uptime_pct: float
    last_reset: str = Field(..., description="Stats reset timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "polymarket",
                "total_requests": 1500,
                "cache_hits": 1080,
                "cache_misses": 420,
                "cache_hit_rate": 0.72,
                "avg_latency_ms": 287.5,
                "errors_count": 5,
                "rate_limit_hits": 0,
                "uptime_pct": 99.67,
                "last_reset": "2025-01-15T00:00:00Z"
            }
        }
