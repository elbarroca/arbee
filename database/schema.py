"""
Database schema and models for prediction market data.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class Event:
    """Event data model matching the API response structure."""
    id: str
    platform: str
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tags: Optional[List[str]] = None
    market_count: Optional[int] = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    total_liquidity: Optional[float] = 0.0
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class Market:
    """Market data model with comprehensive price and metadata."""
    id: str
    platform: str
    event_id: Optional[str] = None
    event_title: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None

    # Price data
    p_yes: Optional[float] = None
    p_no: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

    # Financial metrics
    liquidity: Optional[float] = 0.0
    volume_24h: Optional[float] = 0.0
    total_volume: Optional[float] = 0.0

    # Market mechanics
    num_outcomes: Optional[int] = None

    # Outcome for resolved markets
    outcome: Optional[str] = None

    # Timestamps
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    close_date: Optional[str] = None

    # Raw API response for complete data preservation
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class EventClosed:
    """Closed event data model - identical to Event but for resolved events."""
    id: str
    platform: str
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: str = "closed"  # Always closed for this table
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tags: Optional[List[str]] = None
    market_count: Optional[int] = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    total_liquidity: Optional[float] = 0.0
    closed_at: Optional[str] = None  # When event was moved to closed table
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class MarketClosed:
    """Closed market data model - identical to Market but for resolved markets."""
    id: str
    platform: str
    event_id: Optional[str] = None
    event_title: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    status: str = "closed"  # Always closed for this table

    # Price data at close
    p_yes: Optional[float] = None
    p_no: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

    # Financial metrics at close
    liquidity: Optional[float] = 0.0
    volume_24h: Optional[float] = 0.0
    total_volume: Optional[float] = 0.0

    # Market mechanics
    num_outcomes: Optional[int] = None

    # Outcome for resolved markets
    outcome: Optional[str] = None

    # Timestamps
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    close_date: Optional[str] = None
    closed_at: Optional[str] = None  # When market was moved to closed table

    # Raw API response for complete data preservation
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class ScanSession:
    """Tracks scanning sessions for data lineage."""
    id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    platforms_scanned: Optional[List[str]] = None
    total_events_found: Optional[int] = 0
    total_markets_found: Optional[int] = 0
    total_markets_processed: Optional[int] = 0
    api_keys_used: Optional[int] = 0
    status: str = "running"  # running, completed, failed


