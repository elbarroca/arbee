"""
Trader Scoring Engine
Multi-factor algorithm to identify emerging profitable traders for copy trading.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from scipy import stats
from ..web3.bitquery import BitqueryClient
from ..polymarket import PolymarketClient
from config import settings

logger = logging.getLogger(__name__)


class TraderScoringEngine:
    """
    Multi-factor scoring engine for identifying emerging profitable traders.
    
    Composite Score (0-100) with 5 factors:
    - Early Betting (30%): % trades within 24h of market creation
    - Volume Consistency (20%): Coefficient of variation (lower = better)
    - Win Rate (20%): Success on resolved markets (55%+ = perfect score)
    - Edge Detection (20%): Price movement after bet (60%+ favorable = perfect)
    - Activity Level (10%): Trade frequency (5-20 trades/week ideal)
    
    Filters:
    - Minimum 50 trades in 30 days
    - Active in last 7 days
    - At least 10 resolved markets (for win rate)
    """
    
    def __init__(
        self,
        bitquery_client: Optional[BitqueryClient] = None,
        polymarket_client: Optional[PolymarketClient] = None
    ):
        """
        Initialize trader scoring engine.
        
        Args:
            bitquery_client: Bitquery client for on-chain data
            polymarket_client: Polymarket client for market data
        """
        self.bitquery_client = bitquery_client or BitqueryClient()
        self.polymarket_client = polymarket_client or PolymarketClient()
        
        # Scoring weights (from config)
        self.early_betting_weight = getattr(settings, "TRADER_SCORE_EARLY_BETTING_WEIGHT", 0.30)
        self.volume_consistency_weight = getattr(settings, "TRADER_SCORE_VOLUME_CONSISTENCY_WEIGHT", 0.20)
        self.win_rate_weight = getattr(settings, "TRADER_SCORE_WIN_RATE_WEIGHT", 0.20)
        self.edge_detection_weight = getattr(settings, "TRADER_SCORE_EDGE_DETECTION_WEIGHT", 0.20)
        self.activity_level_weight = getattr(settings, "TRADER_SCORE_ACTIVITY_LEVEL_WEIGHT", 0.10)
        
        # Thresholds
        self.min_trades_30d = getattr(settings, "TRADER_MIN_TRADES_30D", 50)
        self.min_resolved_markets = getattr(settings, "TRADER_MIN_RESOLVED_MARKETS", 10)
        self.win_rate_threshold = getattr(settings, "TRADER_WIN_RATE_THRESHOLD", 0.55)
        self.ideal_weekly_trades_min = getattr(settings, "TRADER_IDEAL_WEEKLY_TRADES_MIN", 5)
        self.ideal_weekly_trades_max = getattr(settings, "TRADER_IDEAL_WEEKLY_TRADES_MAX", 20)
        self.auto_add_score_threshold = getattr(settings, "TRADER_AUTO_ADD_SCORE_THRESHOLD", 70)
        self.auto_pause_score_threshold = 50  # Below this, auto-pause
    
    async def calculate_trader_score(
        self,
        wallet_address: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate composite score for a trader.
        
        Args:
            wallet_address: Wallet address to score
            days: Number of days to analyze
            
        Returns:
            Dict with scoring results:
            - composite_score: Overall score (0-100)
            - early_betting_score: Early betting component (0-30)
            - volume_consistency_score: Volume consistency component (0-20)
            - win_rate_score: Win rate component (0-20)
            - edge_score: Edge detection component (0-20)
            - activity_score: Activity level component (0-10)
            - metrics: Raw metrics used for scoring
            - eligible: Whether trader meets minimum requirements
        """
        # Fetch trader metrics
        metrics = await self.bitquery_client.get_trader_metrics(wallet_address, days=days)
        
        # Check eligibility
        eligible = self._check_eligibility(metrics)
        if not eligible["eligible"]:
            return {
                "wallet_address": wallet_address,
                "composite_score": 0.0,
                "early_betting_score": 0.0,
                "volume_consistency_score": 0.0,
                "win_rate_score": 0.0,
                "edge_score": 0.0,
                "activity_score": 0.0,
                "metrics": metrics,
                "eligible": False,
                "eligibility_reasons": eligible["reasons"]
            }
        
        # Calculate individual scores
        early_betting_score = self._score_early_betting(metrics)
        volume_consistency_score = self._score_volume_consistency(metrics)
        win_rate_score = await self._score_win_rate(wallet_address, metrics)
        edge_score = await self._score_edge_detection(wallet_address, metrics)
        activity_score = self._score_activity_level(metrics)
        
        # Composite score
        composite_score = (
            early_betting_score +
            volume_consistency_score +
            win_rate_score +
            edge_score +
            activity_score
        )
        
        return {
            "wallet_address": wallet_address,
            "composite_score": round(composite_score, 2),
            "early_betting_score": round(early_betting_score, 2),
            "volume_consistency_score": round(volume_consistency_score, 2),
            "win_rate_score": round(win_rate_score, 2),
            "edge_score": round(edge_score, 2),
            "activity_score": round(activity_score, 2),
            "metrics": metrics,
            "eligible": True,
            "should_auto_add": composite_score >= self.auto_add_score_threshold,
            "should_auto_pause": composite_score < self.auto_pause_score_threshold
        }
    
    def _check_eligibility(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if trader meets minimum requirements.
        
        Args:
            metrics: Trader metrics dict
            
        Returns:
            Dict with eligibility status and reasons
        """
        reasons = []
        
        # Check minimum trades
        trade_count = metrics.get("trade_count", 0)
        if trade_count < self.min_trades_30d:
            reasons.append(f"Insufficient trades: {trade_count} < {self.min_trades_30d}")
        
        # Check activity in last 7 days
        transfers = metrics.get("transfers", [])
        if transfers:
            last_trade = max(t.get("timestamp", datetime.min) for t in transfers)
            days_since_last_trade = (datetime.utcnow() - last_trade).days
            if days_since_last_trade > 7:
                reasons.append(f"Inactive: last trade {days_since_last_trade} days ago")
        else:
            reasons.append("No trades found")
        
        return {
            "eligible": len(reasons) == 0,
            "reasons": reasons
        }
    
    def _score_early_betting(self, metrics: Dict[str, Any]) -> float:
        """
        Score early betting factor (0-30 points).
        
        Higher percentage of early bets = higher score.
        """
        early_bet_metrics = metrics.get("early_bet_metrics", {})
        early_bet_pct = early_bet_metrics.get("early_bet_pct", 0.0)
        
        # Linear scaling: 0% = 0 points, 100% = 30 points
        score = (early_bet_pct / 100.0) * (self.early_betting_weight * 100)
        
        return min(score, self.early_betting_weight * 100)
    
    def _score_volume_consistency(self, metrics: Dict[str, Any]) -> float:
        """
        Score volume consistency factor (0-20 points).
        
        Lower coefficient of variation = higher score (more consistent).
        """
        transfers = metrics.get("transfers", [])
        if len(transfers) < 10:
            return 0.0
        
        # Group transfers by day
        daily_volumes: Dict[str, float] = defaultdict(float)
        for transfer in transfers:
            transfer_date = transfer.get("timestamp", datetime.utcnow()).date()
            daily_volumes[str(transfer_date)] += abs(transfer.get("amount", 0.0))
        
        volumes = list(daily_volumes.values())
        if len(volumes) < 3:
            return 0.0
        
        # Calculate coefficient of variation
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        if mean_volume == 0:
            return 0.0
        
        cv = std_volume / mean_volume  # Coefficient of variation
        
        # Lower CV = better consistency
        # Perfect consistency (CV = 0) = 20 points
        # High inconsistency (CV > 1) = 0 points
        # Linear scaling
        score = max(0.0, (1.0 - min(cv, 1.0)) * (self.volume_consistency_weight * 100))
        
        return score
    
    async def _score_win_rate(self, wallet_address: str, metrics: Dict[str, Any]) -> float:
        """
        Score win rate factor (0-20 points).
        
        Requires resolved market data. For now, returns placeholder.
        In production, would query resolved markets and check outcomes.
        """
        # TODO: Implement actual win rate calculation from resolved markets
        # This would require:
        # 1. Query resolved markets from Polymarket
        # 2. Match trader's positions to resolved outcomes
        # 3. Calculate win rate
        
        # Placeholder: return 10 points (middle score) if we have enough trades
        trade_count = metrics.get("trade_count", 0)
        if trade_count >= self.min_resolved_markets:
            # Assume 50% win rate as baseline (would be replaced with actual calculation)
            win_rate = 0.50
            
            # Scale: 0% = 0 points, 55%+ = 20 points
            if win_rate >= self.win_rate_threshold:
                return self.win_rate_weight * 100
            else:
                return (win_rate / self.win_rate_threshold) * (self.win_rate_weight * 100)
        
        return 0.0
    
    async def _score_edge_detection(self, wallet_address: str, metrics: Dict[str, Any]) -> float:
        """
        Score edge detection factor (0-20 points).
        
        Measures price movement after bet (60%+ favorable = perfect score).
        """
        # TODO: Implement actual edge detection
        # This would require:
        # 1. Get trader's buy/sell timestamps
        # 2. Fetch market prices at bet time and 24h later
        # 3. Calculate % of bets with favorable price movement
        
        # Placeholder: return 10 points (middle score)
        transfers = metrics.get("transfers", [])
        if len(transfers) >= 10:
            # Assume 50% favorable movement as baseline
            favorable_pct = 50.0
            
            # Scale: 0% = 0 points, 60%+ = 20 points
            if favorable_pct >= 60.0:
                return self.edge_detection_weight * 100
            else:
                return (favorable_pct / 60.0) * (self.edge_detection_weight * 100)
        
        return 0.0
    
    def _score_activity_level(self, metrics: Dict[str, Any]) -> float:
        """
        Score activity level factor (0-10 points).
        
        Ideal: 5-20 trades/week. Too low or too high = lower score.
        """
        trades_per_week = metrics.get("activity_level", 0.0)
        
        if trades_per_week < self.ideal_weekly_trades_min:
            # Too low: linear scaling from 0 to min
            score = (trades_per_week / self.ideal_weekly_trades_min) * (self.activity_level_weight * 100)
        elif trades_per_week <= self.ideal_weekly_trades_max:
            # Ideal range: full points
            score = self.activity_level_weight * 100
        else:
            # Too high: penalize (but not as harshly)
            excess = trades_per_week - self.ideal_weekly_trades_max
            penalty = min(excess / self.ideal_weekly_trades_max, 0.5)  # Max 50% penalty
            score = (1.0 - penalty) * (self.activity_level_weight * 100)
        
        return min(score, self.activity_level_weight * 100)
    
    async def score_multiple_traders(
        self,
        wallet_addresses: List[str],
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Score multiple traders in batch.
        
        Args:
            wallet_addresses: List of wallet addresses to score
            days: Number of days to analyze
            
        Returns:
            List of scoring results, sorted by composite_score descending
        """
        results = []
        
        for wallet in wallet_addresses:
            try:
                score_result = await self.calculate_trader_score(wallet, days=days)
                results.append(score_result)
            except Exception as e:
                logger.error(f"Error scoring trader {wallet[:8]}...: {e}")
                continue
        
        # Sort by composite score descending
        results.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)
        
        return results

