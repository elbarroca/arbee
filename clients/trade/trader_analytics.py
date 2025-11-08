"""
Trader Analytics Client
Fetches Polymarket Analytics leaderboards and builds copy candidate list.
"""
import logging
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from agents.copy_trading_agent import CopyTrader, CopyTradingAgent
from config import settings

logger = logging.getLogger(__name__)


class TraderAnalyticsClient:
    """
    Client for fetching trader analytics from Polymarket Analytics.
    
    Note: Polymarket Analytics API may not be publicly available.
    This implementation provides a structure that can be adapted when API access is available.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize trader analytics client.
        
        Args:
            api_key: API key for Polymarket Analytics (if available)
        """
        self.api_key = api_key or getattr(settings, "POLYMARKET_ANALYTICS_API_KEY", "")
        self.base_url = "https://analytics.polymarket.com/api"  # Placeholder URL
        
        if not self.api_key:
            logger.warning("Polymarket Analytics API key not configured - using mock data")
    
    async def get_leaderboard(
        self,
        timeframe: str = "30d",
        limit: int = 100,
        min_trades: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Get trader leaderboard.
        
        Args:
            timeframe: Time period ("7d", "30d", "90d", "all")
            limit: Maximum number of traders to return
            min_trades: Minimum trades required
            
        Returns:
            List of trader dicts with metrics
        """
        if not self.api_key:
            # Return mock data for testing
            return self._get_mock_leaderboard(limit)
        
        url = f"{self.base_url}/leaderboard"
        params = {
            "timeframe": timeframe,
            "limit": limit,
            "min_trades": min_trades
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                response.raise_for_status()
                data = response.json()
                return data.get("traders", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch leaderboard: {e}")
            return []
    
    async def get_trader_metrics(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed metrics for a specific trader.
        
        Args:
            wallet_address: Wallet address to query
            
        Returns:
            Dict with trader metrics or None if not found
        """
        if not self.api_key:
            return self._get_mock_trader_metrics(wallet_address)
        
        url = f"{self.base_url}/traders/{wallet_address}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch trader metrics: {e}")
            return None
    
    async def build_copy_candidate_list(
        self,
        copy_agent: CopyTradingAgent,
        timeframe: str = "30d",
        limit: int = 50
    ) -> List[CopyTrader]:
        """
        Build copy candidate list from leaderboard.
        
        Args:
            copy_agent: Copy trading agent to add traders to
            timeframe: Time period for leaderboard
            limit: Maximum traders to evaluate
            
        Returns:
            List of CopyTrader objects added to copy list
        """
        leaderboard = await self.get_leaderboard(timeframe=timeframe, limit=limit)
        
        added_traders = []
        
        for trader_data in leaderboard:
            wallet_address = trader_data.get("wallet_address") or trader_data.get("address")
            if not wallet_address:
                continue
            
            # Calculate metrics
            pnl_30d = float(trader_data.get("pnl_30d", trader_data.get("pnl_30", 0)))
            pnl_90d = float(trader_data.get("pnl_90d", trader_data.get("pnl_90", 0)))
            pnl_all_time = float(trader_data.get("pnl_all_time", trader_data.get("pnl_total", 0)))
            
            trade_count = int(trader_data.get("trade_count", trader_data.get("trades", 0)))
            win_rate = float(trader_data.get("win_rate", trader_data.get("win_rate_pct", 0)) / 100.0)
            
            avg_position_size = float(trader_data.get("avg_position_size", trader_data.get("avg_size", 0)))
            
            # Calculate Sharpe equivalent (simplified)
            sharpe = self._calculate_sharpe_equivalent(trader_data)
            
            # Get categories
            categories = trader_data.get("categories", trader_data.get("markets_traded", []))
            if isinstance(categories, str):
                categories = [categories]
            
            # Calculate wallet age (if available)
            wallet_age_days = trader_data.get("wallet_age_days", 0)
            if not wallet_age_days:
                first_trade = trader_data.get("first_trade_date")
                if first_trade:
                    try:
                        first_dt = datetime.fromisoformat(first_trade.replace('Z', '+00:00'))
                        wallet_age_days = (datetime.utcnow() - first_dt.replace(tzinfo=None)).days
                    except Exception:
                        wallet_age_days = 0
            
            # Create CopyTrader
            trader = CopyTrader(
                wallet_address=wallet_address,
                trader_name=trader_data.get("name") or trader_data.get("username"),
                pnl_30d=pnl_30d,
                pnl_90d=pnl_90d,
                pnl_all_time=pnl_all_time,
                win_rate=win_rate,
                trade_count=trade_count,
                avg_position_size=avg_position_size,
                sharpe_equivalent=sharpe,
                categories_traded=categories,
                wallet_age_days=wallet_age_days,
                last_trade_time=self._parse_last_trade_time(trader_data)
            )
            
            # Add to copy agent if meets criteria
            if copy_agent.add_trader(trader):
                added_traders.append(trader)
        
        logger.info(f"Added {len(added_traders)} traders to copy list from leaderboard")
        return added_traders
    
    def _calculate_sharpe_equivalent(self, trader_data: Dict[str, Any]) -> float:
        """
        Calculate Sharpe ratio equivalent from trader metrics.
        
        Simplified calculation: (avg_return - risk_free_rate) / volatility
        
        Args:
            trader_data: Trader metrics dict
            
        Returns:
            Sharpe ratio equivalent
        """
        # Get return metrics
        pnl_30d = float(trader_data.get("pnl_30d", 0))
        trade_count = int(trader_data.get("trade_count", 1))
        
        if trade_count == 0:
            return 0.0
        
        # Estimate average return per trade
        avg_return = pnl_30d / trade_count if trade_count > 0 else 0.0
        
        # Estimate volatility from win rate (simplified)
        win_rate = float(trader_data.get("win_rate", 0.5))
        volatility = (1 - win_rate) * 0.5  # Simplified volatility estimate
        
        # Risk-free rate
        risk_free_rate = getattr(settings, "RISK_FREE_RATE", 0.02) / 365  # Daily
        
        if volatility == 0:
            return 0.0
        
        sharpe = (avg_return - risk_free_rate) / volatility
        
        # Normalize to reasonable range
        return max(0.0, min(sharpe, 5.0))
    
    def _parse_last_trade_time(self, trader_data: Dict[str, Any]) -> Optional[datetime]:
        """Parse last trade timestamp from trader data"""
        last_trade = trader_data.get("last_trade_date") or trader_data.get("last_trade")
        if last_trade:
            try:
                return datetime.fromisoformat(last_trade.replace('Z', '+00:00'))
            except Exception:
                pass
        return None
    
    def _get_mock_leaderboard(self, limit: int) -> List[Dict[str, Any]]:
        """Generate mock leaderboard data for testing"""
        import random
        
        mock_traders = []
        for i in range(min(limit, 20)):
            mock_traders.append({
                "wallet_address": f"0x{'a' * 40}",
                "name": f"Trader{i+1}",
                "pnl_30d": random.uniform(1000, 50000),
                "pnl_90d": random.uniform(5000, 100000),
                "pnl_all_time": random.uniform(10000, 500000),
                "trade_count": random.randint(200, 1000),
                "win_rate": random.uniform(0.5, 0.8),
                "avg_position_size": random.uniform(100, 5000),
                "categories": ["politics", "sports", "crypto"],
                "wallet_age_days": random.randint(30, 365),
                "last_trade_date": (datetime.utcnow() - timedelta(hours=random.randint(1, 48))).isoformat()
            })
        
        # Sort by PnL descending
        mock_traders.sort(key=lambda x: x["pnl_30d"], reverse=True)
        return mock_traders
    
    def _get_mock_trader_metrics(self, wallet_address: str) -> Dict[str, Any]:
        """Generate mock trader metrics for testing"""
        import random
        
        return {
            "wallet_address": wallet_address,
            "name": f"MockTrader",
            "pnl_30d": random.uniform(1000, 50000),
            "pnl_90d": random.uniform(5000, 100000),
            "pnl_all_time": random.uniform(10000, 500000),
            "trade_count": random.randint(200, 1000),
            "win_rate": random.uniform(0.5, 0.8),
            "avg_position_size": random.uniform(100, 5000),
            "categories": ["politics", "sports"],
            "wallet_age_days": random.randint(30, 365),
            "first_trade_date": (datetime.utcnow() - timedelta(days=random.randint(30, 365))).isoformat(),
            "last_trade_date": (datetime.utcnow() - timedelta(hours=random.randint(1, 48))).isoformat()
        }
