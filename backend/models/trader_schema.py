"""
Database schema for trader analytics enhancements.

Defines dataclasses for:
- Open positions (unrealized PnL)
- Radar chart metrics
- Tag-specific credibility scores
- Aggregated market position details
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class EliteOpenPosition:
    """Current elite open position with unrealized PnL."""

    id: str  # Primary key: proxy_wallet + condition_id + outcome_index
    proxy_wallet: str  # Foreign key to wallets
    event_id: Optional[str] = None  # Foreign key to events
    condition_id: str = None  # Market condition ID
    asset: str = None  # ERC-1155 token ID

    # Position details
    outcome: Optional[str] = None
    outcome_index: Optional[int] = None
    size: float = 0.0  # Current position size
    avg_entry_price: float = 0.0  # Average entry price
    current_price: float = 0.0  # Current market price
    unrealized_pnl: float = 0.0  # (current_price - avg_entry_price) × size
    position_value: float = 0.0  # size × current_price (USDC value)

    # Entry info
    first_trade_ts: Optional[int] = None  # When position was first opened
    last_trade_ts: Optional[int] = None  # Most recent trade in this position

    # Market metadata
    title: Optional[str] = None
    slug: Optional[str] = None
    event_slug: Optional[str] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class EliteTraderPerformance:
    """Elite trader performance metrics vs market average, organized by trade end date."""

    id: str  # Primary key: proxy_wallet + event_id + end_date
    proxy_wallet: str  # Foreign key to wallets
    event_id: str  # Foreign key to events (ensures canonical organization)
    event_title: Optional[str] = None
    event_category: Optional[str] = None

    # Time-based organization (canonical)
    trade_end_date: datetime  # When the market resolved (for chronological sorting)
    analysis_period_days: int = 30  # Lookback period for market average calculation

    # Trader performance metrics
    trader_total_positions: int = 0  # Number of positions in this event
    trader_total_volume: float = 0.0  # Total USDC traded
    trader_realized_pnl: float = 0.0  # Total realized PnL
    trader_roi: float = 0.0  # Return on investment (PnL / volume)
    trader_win_rate: float = 0.0  # Percentage of winning positions
    trader_avg_position_size: float = 0.0  # Average position size
    trader_entry_timing_score: float = 0.0  # How early/late trader entered vs market

    # Market average metrics (benchmark)
    market_total_positions: int = 0  # Total positions by all traders in event
    market_total_volume: float = 0.0  # Total market volume
    market_avg_roi: float = 0.0  # Average ROI across all traders
    market_win_rate: float = 0.0  # Market-wide win rate
    market_avg_position_size: float = 0.0  # Average position size across market
    market_participation_rate: float = 0.0  # Percentage of market volume by trader

    # Performance vs market (key metrics)
    roi_vs_market: float = 0.0  # trader_roi - market_avg_roi
    win_rate_vs_market: float = 0.0  # trader_win_rate - market_win_rate
    volume_vs_market_percentile: float = 0.0  # Trader's volume percentile in market
    timing_vs_market_percentile: float = 0.0  # Entry timing percentile

    # Canonical performance score (composite)
    canonical_performance_score: float = 0.0  # Weighted score combining all metrics

    # Activity metrics
    trader_rank_in_event: int = 0  # Rank by volume in this event
    total_traders_in_event: int = 0  # Total number of traders in event

    # Metadata
    computed_at: datetime  # When this analysis was computed
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class WalletRadarMetrics:
    """Radar chart metrics for trader profiling."""

    proxy_wallet: str  # Primary key

    # Radar chart metrics (normalized 0-1)
    unique_markets_score: float = 0.0  # Diversity of markets traded
    trade_correlation_score: float = 0.0  # Alignment with smart money
    entry_timing_score: float = 0.0  # Early vs late entry percentile
    position_size_score: float = 0.0  # Relative position sizing
    wallet_tx_delta_score: float = 0.0  # Profitability ratio (PnL/volume)

    # Raw values for reference
    unique_markets_count: int = 0
    avg_correlation: float = 0.0  # Average correlation with Tier A wallets
    avg_entry_rank: float = 0.0  # Average entry timing percentile
    avg_size_rank: float = 0.0  # Average position size percentile

    computed_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class TraderTagCredibility:
    """Tag-specific credibility score for trader."""

    id: str  # Primary key: proxy_wallet + tag
    proxy_wallet: str  # Foreign key to wallets
    tag: str  # Category or tag from events

    # Tag-specific credibility
    credibility_score: float = 0.0  # Composite score (0-1)
    tag_roi: float = 0.0  # ROI within this tag
    tag_win_rate: float = 0.0  # Win rate within this tag
    tag_volume: float = 0.0  # Total volume in this tag
    tag_positions: int = 0  # Number of positions in this tag
    tag_rank: Optional[int] = None  # Rank among all traders in this tag

    # Percentile ranks for composite scoring
    roi_percentile: float = 0.0  # ROI percentile within tag
    win_rate_percentile: float = 0.0  # Win rate percentile within tag
    volume_percentile: float = 0.0  # Volume percentile within tag

    computed_at: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class TraderMarketPosition:
    """Aggregated position details for trader in specific market.

    This is a computed/aggregated view, not a direct database table.
    Built from trades and open_positions data.
    """

    proxy_wallet: str
    event_id: Optional[str] = None
    condition_id: str = None

    # Trade volume breakdown
    buy_volume: float = 0.0  # Total USDC spent buying
    sell_volume: float = 0.0  # Total USDC received selling
    net_volume: float = 0.0  # buy_volume - sell_volume

    # Price metrics
    avg_buy_price: float = 0.0  # Average price paid
    avg_sell_price: float = 0.0  # Average price received
    init_price: float = 0.0  # Price when first entered
    current_price: float = 0.0  # Current market price

    # Position details
    current_position_size: float = 0.0  # Net position (tokens)
    position_value: float = 0.0  # current_position_size × current_price

    # PnL
    realized_pnl: float = 0.0  # PnL from closed portion
    unrealized_pnl: float = 0.0  # PnL from open portion
    total_pnl: float = 0.0  # realized + unrealized

    # Activity
    market_trades_count: int = 0  # Number of trades in this market
    first_trade_ts: Optional[int] = None
    last_trade_ts: Optional[int] = None

    # Market concentration
    market_volume: float = 0.0  # Total market volume
    volume_share: float = 0.0  # Trader's volume / market volume

    # Market metadata
    market_title: Optional[str] = None
    market_slug: Optional[str] = None
    event_slug: Optional[str] = None
    outcome: Optional[str] = None

    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class TraderProfile:
    """Complete trader profile (aggregated view for API responses).

    Combines data from multiple tables for detailed trader view.
    """

    proxy_wallet: str
    name: Optional[str] = None
    pseudonym: Optional[str] = None
    bio: Optional[str] = None
    profile_image: Optional[str] = None

    # Activity summary
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    total_trades: int = 0
    total_markets: int = 0

    # Global stats
    total_volume: float = 0.0
    realized_pnl: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    n_positions: int = 0

    # Time-windowed PnL
    pnl_30d: float = 0.0
    pnl_90d: float = 0.0
    pnl_all_time: float = 0.0

    # Open positions
    open_positions_count: int = 0
    open_positions_value: float = 0.0
    unrealized_pnl: float = 0.0

    # Metrics
    wallet_tx_delta_ratio: float = 0.0  # PnL / volume ratio

    # Eligibility
    is_eligible: bool = False
    tier: Optional[str] = None  # A/B/C

    # Radar metrics (optional)
    radar_metrics: Optional[WalletRadarMetrics] = None

    # Tag credibility (optional list)
    tag_credibility: Optional[List[TraderTagCredibility]] = field(default_factory=list)

    # Market positions (optional list)
    market_positions: Optional[List[TraderMarketPosition]] = field(default_factory=list)

    # Open positions (optional list)
    open_positions: Optional[List[EliteOpenPosition]] = field(default_factory=list)

    raw_data: Optional[Dict[str, Any]] = None


# Type aliases for convenience
EliteOpenPositionList = List[EliteOpenPosition]
EliteTraderPerformanceList = List[EliteTraderPerformance]
RadarMetricsList = List[WalletRadarMetrics]
TagCredibilityList = List[TraderTagCredibility]
MarketPositionList = List[TraderMarketPosition]
