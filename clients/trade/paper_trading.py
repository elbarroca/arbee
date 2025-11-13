"""
Paper Trading Logger
Simulates trades with realistic slippage models and tracks P&L.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import pandas as pd
import numpy as np

try:
    import quantstats as qs
    import empyrical as ep
    HAS_PERFORMANCE_LIBS = True
except ImportError:
    HAS_PERFORMANCE_LIBS = False

from betting.copy_trading import TradeSignal
from database.client import SupabaseClient
from ..polymarket import PolymarketClient
from config import settings

logger = logging.getLogger(__name__)

if not HAS_PERFORMANCE_LIBS:
    logger.warning("quantstats/empyrical not installed - performance metrics will be simplified")


class SlippageModel(str, Enum):
    """Slippage model types"""
    FIXED = "fixed"
    VOLUME_BASED = "volume_based"
    SPREAD_BASED = "spread_based"


class PaperTradingLogger:
    """
    Paper trading logger for simulating trades with realistic slippage.
    
    Features:
    - 3 slippage models: fixed, volume-based, spread-based
    - FIFO position tracking with cost basis
    - Mark-to-market P&L calculation
    - Performance metrics (Sharpe, Sortino, drawdown, win rate)
    """
    
    def __init__(
        self,
        db_client: Optional[SupabaseClient] = None,
        polymarket_client: Optional[PolymarketClient] = None,
        slippage_model: SlippageModel = SlippageModel.FIXED,
        fixed_slippage_bps: int = 50  # 0.5% default
    ):
        """
        Initialize paper trading logger.
        
        Args:
            db_client: Supabase database client
            polymarket_client: Polymarket client for market data
            slippage_model: Slippage model to use
            fixed_slippage_bps: Fixed slippage in basis points (for FIXED model)
        """
        self.db_client = db_client or SupabaseClient()
        self.polymarket_client = polymarket_client or PolymarketClient()
        self.slippage_model = slippage_model
        self.fixed_slippage_bps = fixed_slippage_bps
        
        # FIFO position tracking: market_slug -> List[positions]
        # Each position: {amount, cost_basis, timestamp}
        self._positions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Trade history
        self._trade_history: List[Dict[str, Any]] = []
    
    async def log_trade(
        self,
        signal: TradeSignal,
        expected_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Log a simulated trade.
        
        Args:
            signal: Trade signal to simulate
            expected_price: Expected fill price (if None, uses signal.price)
            
        Returns:
            Dict with trade execution details:
            - fill_price: Simulated fill price
            - slippage_bps: Slippage in basis points
            - status: filled or rejected
        """
        market_slug = signal.market_slug
        side = signal.side
        size_usd = signal.size_usd
        expected = expected_price or signal.price
        
        # Calculate fill price based on slippage model
        fill_price, slippage_bps = await self._calculate_fill_price(
            market_slug=market_slug,
            side=side,
            expected_price=expected,
            size_usd=size_usd
        )
        
        # Check if trade should be rejected (e.g., slippage too high)
        max_slippage_bps = getattr(settings, "COPY_TRADING_MAX_SLIPPAGE_BPS", 50)
        if slippage_bps > max_slippage_bps:
            status = "rejected"
            logger.debug(f"Trade rejected: slippage {slippage_bps} bps > max {max_slippage_bps} bps")
        else:
            status = "filled"
            
            # Update FIFO positions
            self._update_positions(market_slug, side, size_usd, fill_price)
        
        # Create trade record
        trade_record = {
            "signal_id": signal.transaction_hash,
            "wallet_address": signal.wallet_address,
            "market_slug": market_slug,
            "side": side,
            "size_usd": size_usd,
            "expected_price": expected,
            "fill_price": fill_price,
            "slippage_bps": slippage_bps,
            "status": status,
            "timestamp": signal.timestamp or datetime.utcnow()
        }
        
        self._trade_history.append(trade_record)
        
        # Save to database
        try:
            await self.db_client.insert_paper_trading_log(
                signal_id=signal.transaction_hash,
                wallet_address=signal.wallet_address,
                market_slug=market_slug,
                side=side,
                size_usd=size_usd,
                expected_price=expected,
                fill_price=fill_price,
                slippage_bps=slippage_bps,
                status=status
            )
        except Exception as e:
            logger.error(f"Error saving paper trading log to database: {e}")
        
        return trade_record
    
    async def _calculate_fill_price(
        self,
        market_slug: str,
        side: str,
        expected_price: float,
        size_usd: float
    ) -> tuple[float, int]:
        """
        Calculate fill price based on slippage model.
        
        Returns:
            Tuple of (fill_price, slippage_bps)
        """
        if self.slippage_model == SlippageModel.FIXED:
            # Fixed slippage: add/subtract fixed percentage
            slippage_pct = self.fixed_slippage_bps / 10000.0
            if side == "BUY":
                fill_price = expected_price * (1 + slippage_pct)
            else:  # SELL
                fill_price = expected_price * (1 - slippage_pct)
            
            slippage_bps = self.fixed_slippage_bps
        
        elif self.slippage_model == SlippageModel.VOLUME_BASED:
            # Volume-based: Zipline-style impact model
            # Impact = k * (size / volume) ^ alpha
            # For now, simplified: impact increases with size
            market = await self.polymarket_client.gamma.get_market(market_slug)
            volume = market.get("volumeNum", 0) or market.get("volume", 0)
            
            if volume > 0:
                # Impact factor: larger trades = more slippage
                impact_factor = min((size_usd / volume) ** 0.5, 0.02)  # Max 2% impact
            else:
                raise ValueError(f"No volume data available for market {market_slug} - cannot calculate volume-based slippage")
            
            if side == "BUY":
                fill_price = expected_price * (1 + impact_factor)
            else:
                fill_price = expected_price * (1 - impact_factor)
            
            slippage_bps = int(impact_factor * 10000)
        
        elif self.slippage_model == SlippageModel.SPREAD_BASED:
            # Spread-based: use orderbook spread
            market = await self.polymarket_client.gamma.get_market(market_slug)
            token_ids = market.get("clobTokenIds", [])
            
            if not token_ids or len(token_ids) < 2:
                raise ValueError(f"Insufficient token IDs for market {market_slug} - cannot calculate spread-based slippage")
            
            token_id = token_ids[1] if side == "BUY" else token_ids[0]
            orderbook = self.polymarket_client.clob.get_orderbook(token_id, depth=10)
            
            best_bid = orderbook.get("best_bid", 0)
            best_ask = orderbook.get("best_ask", 1)
            
            if side == "BUY":
                fill_price = best_ask  # Take ask
            else:
                fill_price = best_bid  # Take bid
            
            mid_price = (best_bid + best_ask) / 2
            if mid_price > 0:
                slippage_bps = int(abs(fill_price - mid_price) / mid_price * 10000)
            else:
                raise ValueError(f"Invalid orderbook prices (bid={best_bid}, ask={best_ask}) for market {market_slug}")
        
        else:
            # Default to fixed
            slippage_pct = self.fixed_slippage_bps / 10000.0
            fill_price = expected_price * (1 + slippage_pct) if side == "BUY" else expected_price * (1 - slippage_pct)
            slippage_bps = self.fixed_slippage_bps
        
        return fill_price, slippage_bps
    
    def _update_positions(
        self,
        market_slug: str,
        side: str,
        size_usd: float,
        fill_price: float
    ):
        """
        Update FIFO positions after a trade.
        
        Args:
            market_slug: Market identifier
            side: BUY or SELL
            size_usd: Trade size in USD
            fill_price: Fill price
        """
        # Calculate token amount (simplified: assume 1 token = $1 at price)
        token_amount = size_usd / fill_price if fill_price > 0 else 0
        
        if side == "BUY":
            # Add to positions (FIFO queue)
            self._positions[market_slug].append({
                "amount": token_amount,
                "cost_basis": fill_price,
                "timestamp": datetime.utcnow()
            })
        else:  # SELL
            # Remove from positions (FIFO)
            remaining_sell = token_amount
            
            while remaining_sell > 0 and self._positions[market_slug]:
                position = self._positions[market_slug][0]
                
                if position["amount"] <= remaining_sell:
                    # Fully consumed position
                    remaining_sell -= position["amount"]
                    self._positions[market_slug].pop(0)
                else:
                    # Partially consumed position
                    position["amount"] -= remaining_sell
                    remaining_sell = 0
    
    async def mark_to_market(
        self,
        market_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update unrealized P&L for all open positions.
        
        Args:
            market_prices: Dict mapping market_slug to current price
            
        Returns:
            Dict mapping market_slug to unrealized P&L
        """
        unrealized_pnl: Dict[str, float] = {}
        
        for market_slug, positions in self._positions.items():
            current_price = market_prices.get(market_slug)
            if current_price is None:
                continue
            
            total_unrealized = 0.0
            for position in positions:
                # Unrealized P&L = (current_price - cost_basis) * amount
                pnl = (current_price - position["cost_basis"]) * position["amount"]
                total_unrealized += pnl
            
            unrealized_pnl[market_slug] = total_unrealized
        
        return unrealized_pnl
    
    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get current positions with cost basis.
        
        Returns:
            Dict mapping market_slug to list of positions
        """
        return dict(self._positions)
    
    def get_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics using quantstats and empyrical.
        
        Args:
            start_date: Start date for metrics calculation
            end_date: End date for metrics calculation
            
        Returns:
            Dict with performance metrics:
            - sharpe_ratio: Sharpe ratio (annualized)
            - sortino_ratio: Sortino ratio
            - max_drawdown: Maximum drawdown percentage
            - win_rate: Win rate (percentage)
            - total_pnl: Total P&L
            - total_trades: Total number of trades
        """
        # Filter trades by date range
        filtered_trades = self._trade_history
        if start_date:
            filtered_trades = [t for t in filtered_trades if t["timestamp"] >= start_date]
        if end_date:
            filtered_trades = [t for t in filtered_trades if t["timestamp"] <= end_date]
        
        filled_trades = [t for t in filtered_trades if t["status"] == "filled"]
        
        if not filled_trades:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_trades": 0
            }
        
        # Calculate trade P&L
        trade_pnls = []
        for trade in filled_trades:
            # Simplified P&L calculation
            trade_pnl = (trade["fill_price"] - trade["expected_price"]) * trade["size_usd"] * (
                1 if trade["side"] == "BUY" else -1
            )
            trade_pnls.append({
                "timestamp": trade["timestamp"],
                "pnl": trade_pnl
            })
        
        total_pnl = sum(t["pnl"] for t in trade_pnls)
        winning_trades = len([t for t in trade_pnls if t["pnl"] > 0])
        win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0.0
        
        # Calculate daily returns for Sharpe/Sortino/Drawdown
        if HAS_PERFORMANCE_LIBS and trade_pnls:
            # Create DataFrame with daily returns
            df = pd.DataFrame(trade_pnls)
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            daily_returns = df.groupby("date")["pnl"].sum()
            
            # Convert to percentage returns (simplified - assumes $10k starting capital)
            starting_capital = 10000.0
            cumulative_pnl = daily_returns.cumsum()
            equity = starting_capital + cumulative_pnl
            daily_pct_returns = equity.pct_change().fillna(0)
            
            # Calculate metrics
            sharpe_ratio = ep.sharpe_ratio(daily_pct_returns, annualization=252) if len(daily_pct_returns) > 1 else 0.0
            sortino_ratio = ep.sortino_ratio(daily_pct_returns, annualization=252) if len(daily_pct_returns) > 1 else 0.0
            max_drawdown = ep.max_drawdown(daily_pct_returns) if len(daily_pct_returns) > 1 else 0.0
            
            # Handle NaN values
            sharpe_ratio = sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0
            sortino_ratio = sortino_ratio if not np.isnan(sortino_ratio) else 0.0
            max_drawdown = abs(max_drawdown) if not np.isnan(max_drawdown) else 0.0
        else:
            # Simplified metrics without quantstats/empyrical
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
        
        return {
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_trades": len(filled_trades),
            "winning_trades": winning_trades,
            "losing_trades": len(filled_trades) - winning_trades
        }
    
    def get_equity_curve(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get equity curve (cumulative P&L over time).
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dicts with timestamp and cumulative_pnl
        """
        filtered_trades = self._trade_history
        if start_date:
            filtered_trades = [t for t in filtered_trades if t["timestamp"] >= start_date]
        if end_date:
            filtered_trades = [t for t in filtered_trades if t["timestamp"] <= end_date]
        
        filled_trades = sorted(
            [t for t in filtered_trades if t["status"] == "filled"],
            key=lambda x: x["timestamp"]
        )
        
        equity_curve = []
        cumulative_pnl = 0.0
        
        for trade in filled_trades:
            # Calculate trade P&L (simplified)
            trade_pnl = (trade["fill_price"] - trade["expected_price"]) * trade["size_usd"] * (
                1 if trade["side"] == "BUY" else -1
            )
            cumulative_pnl += trade_pnl
            
            equity_curve.append({
                "timestamp": trade["timestamp"],
                "cumulative_pnl": cumulative_pnl
            })
        
        return equity_curve

