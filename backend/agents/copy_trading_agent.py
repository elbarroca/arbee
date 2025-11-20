"""
Copy Trading Agent
Identifies and manages copy candidate list from successful traders.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from config import settings

logger = logging.getLogger(__name__)


class CopyTrader(BaseModel):
    """Represents a trader to copy"""
    wallet_address: str = Field(..., description="Wallet address")
    trader_name: Optional[str] = Field(None, description="Optional trader name/identifier")
    pnl_30d: float = Field(0.0, description="30-day PnL")
    pnl_90d: float = Field(0.0, description="90-day PnL")
    pnl_all_time: float = Field(0.0, description="All-time PnL")
    win_rate: float = Field(0.0, description="Win rate (0-1)")
    trade_count: int = Field(0, description="Total number of trades")
    avg_position_size: float = Field(0.0, description="Average position size in USD")
    sharpe_equivalent: float = Field(0.0, description="Sharpe ratio equivalent")
    categories_traded: List[str] = Field(default_factory=list, description="Market categories")
    wallet_age_days: int = Field(0, description="Wallet age in days")
    added_date: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field("active", description="Status: active, paused, removed")
    last_trade_time: Optional[datetime] = Field(None, description="Last trade timestamp")


class CopyTradingAgent:
    """
    Agent for identifying and managing copy trading candidates.
    
    Filters traders by:
    - 30-day positive PnL
    - Sharpe > 0.7 equivalent
    - Min 200 trades
    - Wallet age < N days (configurable)
    """
    
    def __init__(
        self,
        min_pnl_30d: float = 0.0,
        min_sharpe: float = 0.7,
        min_trades: int = 200,
        max_wallet_age_days: Optional[int] = None,
        min_win_rate: float = 0.5
    ):
        """
        Initialize copy trading agent.
        
        Args:
            min_pnl_30d: Minimum 30-day PnL to qualify
            min_sharpe: Minimum Sharpe ratio equivalent
            min_trades: Minimum number of trades
            max_wallet_age_days: Maximum wallet age in days (None = no limit)
            min_win_rate: Minimum win rate (0-1)
        """
        self.min_pnl_30d = min_pnl_30d
        self.min_sharpe = min_sharpe
        self.min_trades = min_trades
        self.max_wallet_age_days = max_wallet_age_days
        self.min_win_rate = min_win_rate
        
        self.copy_list: Dict[str, CopyTrader] = {}
    
    def add_trader(self, trader: CopyTrader) -> bool:
        """
        Add a trader to the copy list if they meet criteria.
        
        Args:
            trader: Trader to evaluate and potentially add
            
        Returns:
            True if added, False if rejected
        """
        # Check criteria
        if trader.pnl_30d < self.min_pnl_30d:
            logger.debug(f"Trader {trader.wallet_address[:8]}... rejected: PnL too low")
            return False
        
        if trader.sharpe_equivalent < self.min_sharpe:
            logger.debug(f"Trader {trader.wallet_address[:8]}... rejected: Sharpe too low")
            return False
        
        if trader.trade_count < self.min_trades:
            logger.debug(f"Trader {trader.wallet_address[:8]}... rejected: Too few trades")
            return False
        
        if trader.win_rate < self.min_win_rate:
            logger.debug(f"Trader {trader.wallet_address[:8]}... rejected: Win rate too low")
            return False
        
        if self.max_wallet_age_days and trader.wallet_age_days > self.max_wallet_age_days:
            logger.debug(f"Trader {trader.wallet_address[:8]}... rejected: Wallet too old")
            return False
        
        # Add to copy list
        self.copy_list[trader.wallet_address.lower()] = trader
        logger.info(f"Added trader to copy list: {trader.wallet_address[:8]}... (PnL: ${trader.pnl_30d:,.0f})")
        return True
    
    def remove_trader(self, wallet_address: str) -> bool:
        """Remove a trader from the copy list"""
        addr = wallet_address.lower()
        if addr in self.copy_list:
            trader = self.copy_list[addr]
            trader.status = "removed"
            del self.copy_list[addr]
            logger.info(f"Removed trader from copy list: {wallet_address[:8]}...")
            return True
        return False
    
    def pause_trader(self, wallet_address: str) -> bool:
        """Pause copying a trader (temporarily)"""
        addr = wallet_address.lower()
        if addr in self.copy_list:
            self.copy_list[addr].status = "paused"
            logger.info(f"Paused copying trader: {wallet_address[:8]}...")
            return True
        return False
    
    def resume_trader(self, wallet_address: str) -> bool:
        """Resume copying a trader"""
        addr = wallet_address.lower()
        if addr in self.copy_list:
            self.copy_list[addr].status = "active"
            logger.info(f"Resumed copying trader: {wallet_address[:8]}...")
            return True
        return False
    
    def get_active_traders(self) -> List[CopyTrader]:
        """Get list of active traders to copy"""
        return [
            trader for trader in self.copy_list.values()
            if trader.status == "active"
        ]
    
    def is_trader_active(self, wallet_address: str) -> bool:
        """Check if a trader is in the active copy list"""
        addr = wallet_address.lower()
        trader = self.copy_list.get(addr)
        return trader is not None and trader.status == "active"
    
    def update_trader_metrics(
        self,
        wallet_address: str,
        **metrics
    ) -> bool:
        """
        Update trader metrics (called periodically).
        
        Args:
            wallet_address: Wallet address
            **metrics: Metric updates (pnl_30d, win_rate, etc.)
        """
        addr = wallet_address.lower()
        if addr not in self.copy_list:
            return False
        
        trader = self.copy_list[addr]
        for key, value in metrics.items():
            if hasattr(trader, key):
                setattr(trader, key, value)
        
        # Re-evaluate if still meets criteria
        if not self._meets_criteria(trader):
            logger.warning(f"Trader {wallet_address[:8]}... no longer meets criteria, pausing")
            trader.status = "paused"
            return False
        
        return True
    
    def _meets_criteria(self, trader: CopyTrader) -> bool:
        """Check if trader meets all criteria"""
        return (
            trader.pnl_30d >= self.min_pnl_30d and
            trader.sharpe_equivalent >= self.min_sharpe and
            trader.trade_count >= self.min_trades and
            trader.win_rate >= self.min_win_rate and
            (self.max_wallet_age_days is None or trader.wallet_age_days <= self.max_wallet_age_days)
        )
    
    def get_copy_list_summary(self) -> Dict[str, Any]:
        """Get summary of copy list"""
        active = self.get_active_traders()
        return {
            "total_traders": len(self.copy_list),
            "active_traders": len(active),
            "paused_traders": len([t for t in self.copy_list.values() if t.status == "paused"]),
            "total_pnl_30d": sum(t.pnl_30d for t in active),
            "avg_sharpe": sum(t.sharpe_equivalent for t in active) / len(active) if active else 0.0,
            "traders": [
                {
                    "wallet_address": t.wallet_address,
                    "pnl_30d": t.pnl_30d,
                    "sharpe": t.sharpe_equivalent,
                    "trade_count": t.trade_count,
                    "status": t.status
                }
                for t in active
            ]
        }

