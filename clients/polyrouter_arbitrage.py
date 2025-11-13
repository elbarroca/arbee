"""
polyrouter_arbitrage.py

Advanced arbitrage detection for prediction markets and sports betting.

This module implements four types of arbitrage strategies:
1. Cross-Platform Arbitrage: Buy YES on platform A, NO on platform B
2. Single-Platform Mispricing: YES + NO ≠ 1.0 on same platform
3. Multi-Leg Arbitrage: 3+ outcome markets with guaranteed profit
4. Threshold Market Arbitrage: Related markets (e.g., >50% vs >60%)

Public API:
- find_cross_platform_arbitrage(markets, threshold)
- find_single_platform_mispricing(markets, threshold)
- find_multi_leg_arbitrage(markets, threshold)
- find_threshold_arbitrage(markets, threshold)
- calculate_kelly_sizing(opportunity, bankroll)
- calculate_expected_value(opportunity)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import math

__all__ = [
    "ArbitrageOpportunity",
    "find_cross_platform_arbitrage",
    "find_single_platform_mispricing",
    "find_multi_leg_arbitrage",
    "find_threshold_arbitrage",
    "calculate_kelly_sizing",
    "calculate_expected_value",
    "rank_opportunities",
]

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Structured representation of an arbitrage opportunity."""

    type: str  # "cross_platform", "single_platform", "multi_leg", "threshold"
    event_name: str
    markets: List[Dict[str, Any]]
    legs: List[Dict[str, Any]]  # List of bets to place
    margin: float  # Profit margin (0-1)
    margin_percent: float  # Profit margin as percentage
    total_stake_ratio: float  # Total required stake relative to payout
    expected_roi: float  # Return on investment
    risk_score: float  # Risk assessment (0-1, lower is better)
    liquidity_score: float  # Liquidity assessment (0-1, higher is better)
    execution_complexity: str  # "simple", "moderate", "complex"
    estimated_slippage: float  # Expected slippage (0-1)
    post_fee_margin: float  # Margin after fees
    recommended_stake: Optional[float] = None  # Kelly-sized stake
    kelly_fraction: Optional[float] = None  # Kelly fraction used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "event_name": self.event_name,
            "markets": self.markets,
            "legs": self.legs,
            "margin": self.margin,
            "margin_percent": self.margin_percent,
            "total_stake_ratio": self.total_stake_ratio,
            "expected_roi": self.expected_roi,
            "risk_score": self.risk_score,
            "liquidity_score": self.liquidity_score,
            "execution_complexity": self.execution_complexity,
            "estimated_slippage": self.estimated_slippage,
            "post_fee_margin": self.post_fee_margin,
            "recommended_stake": self.recommended_stake,
            "kelly_fraction": self.kelly_fraction,
        }


# ==================== Helper Functions ==================== #

def _extract_price(price_data: Any, side: str) -> Optional[float]:
    """Extract YES or NO price from various price data formats."""
    if isinstance(price_data, dict):
        # Try multiple keys
        for key in [side, side.upper(), side.capitalize()]:
            if key in price_data:
                val = price_data[key]
                if isinstance(val, dict) and "price" in val:
                    return float(val["price"])
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
    elif isinstance(price_data, (int, float)):
        return float(price_data)
    return None


def _calculate_liquidity_score(markets: List[Dict[str, Any]]) -> float:
    """Calculate liquidity score based on volume and available liquidity."""
    total_liquidity = sum(m.get("liquidity", 0) for m in markets)
    total_volume = sum(m.get("volume_total", m.get("volume", 0)) for m in markets)

    # Normalize scores
    liquidity_score = min(total_liquidity / 50000, 1.0)  # $50k = full score
    volume_score = min(total_volume / 100000, 1.0)  # $100k = full score

    # Weighted average
    return liquidity_score * 0.6 + volume_score * 0.4


def _calculate_risk_score(opportunity_type: str, margin: float, liquidity_score: float) -> float:
    """Calculate risk score for an opportunity (lower is better)."""
    # Base risk by type
    type_risk = {
        "cross_platform": 0.3,  # Medium risk (execution across platforms)
        "single_platform": 0.2,  # Lower risk (single execution)
        "multi_leg": 0.5,  # Higher risk (multiple legs)
        "threshold": 0.4,  # Moderate risk (correlation risk)
    }.get(opportunity_type, 0.5)

    # Margin risk (lower margin = higher risk)
    margin_risk = max(0, 1.0 - (margin / 0.10))  # 10% margin = no margin risk

    # Liquidity risk
    liquidity_risk = 1.0 - liquidity_score

    # Weighted combination
    risk_score = (
        type_risk * 0.4 +
        margin_risk * 0.4 +
        liquidity_risk * 0.2
    )

    return min(risk_score, 1.0)


def _estimate_fees(markets: List[Dict[str, Any]]) -> float:
    """Estimate total transaction fees."""
    total_fees = 0.0
    for market in markets:
        fee_rate = market.get("fee_rate") or 0.02  # Default 2%
        total_fees += fee_rate
    return total_fees / len(markets) if markets else 0.02


def _estimate_slippage(liquidity: float, stake: float) -> float:
    """Estimate slippage based on liquidity and stake size."""
    if liquidity <= 0:
        return 0.05  # 5% default slippage for illiquid markets

    # Simple model: slippage proportional to stake/liquidity ratio
    ratio = stake / liquidity

    if ratio < 0.01:
        return 0.001  # 0.1% for small orders
    elif ratio < 0.05:
        return 0.005  # 0.5% for medium orders
    elif ratio < 0.10:
        return 0.01  # 1% for large orders
    else:
        return 0.03  # 3% for very large orders


# ==================== Cross-Platform Arbitrage ==================== #

def find_cross_platform_arbitrage(
    markets: List[Dict[str, Any]],
    threshold: float = 0.01,
    min_liquidity: float = 1000.0,
) -> List[ArbitrageOpportunity]:
    """Find cross-platform arbitrage: Buy YES on platform A, NO on platform B.

    Parameters
    ----------
    markets : List[Dict[str, Any]]
        List of markets to analyze
    threshold : float
        Minimum arbitrage margin required. Default: 1%
    min_liquidity : float
        Minimum liquidity per market. Default: $1000

    Returns
    -------
    List[ArbitrageOpportunity]
        List of arbitrage opportunities sorted by margin
    """
    # Group markets by event
    events = defaultdict(list)
    for market in markets:
        event_key = (
            market.get("event_name") or
            market.get("title") or
            market.get("question") or
            market.get("event_id")
        )
        if event_key:
            events[event_key].append(market)

    opportunities = []

    for event_name, event_markets in events.items():
        # Filter by liquidity
        liquid_markets = [
            m for m in event_markets
            if m.get("liquidity", 0) >= min_liquidity
        ]

        if len(liquid_markets) < 2:
            continue

        # Find best YES and NO prices across platforms
        best_yes_price = None
        best_yes_market = None
        best_no_price = None
        best_no_market = None

        for market in liquid_markets:
            platform = market.get("platform") or market.get("provider")
            prices = market.get("current_prices") or {}

            yes_price = _extract_price(prices, "yes")
            no_price = _extract_price(prices, "no")

            if yes_price is not None:
                if best_yes_price is None or yes_price < best_yes_price:
                    best_yes_price = yes_price
                    best_yes_market = market

            if no_price is not None:
                if best_no_price is None or no_price < best_no_price:
                    best_no_price = no_price
                    best_no_market = market

        # Check for arbitrage
        if best_yes_price and best_no_price and best_yes_market and best_no_market:
            total_cost = best_yes_price + best_no_price
            margin = 1.0 - total_cost

            if margin > threshold:
                # Calculate metrics
                involved_markets = [best_yes_market, best_no_market]
                liquidity_score = _calculate_liquidity_score(involved_markets)
                risk_score = _calculate_risk_score("cross_platform", margin, liquidity_score)

                # Estimate fees and slippage
                avg_fee = _estimate_fees(involved_markets)
                total_liquidity = sum(m.get("liquidity", 0) for m in involved_markets)
                est_slippage = _estimate_slippage(total_liquidity, 1000)  # Estimate for $1k trade

                post_fee_margin = margin - avg_fee - est_slippage

                if post_fee_margin > 0:
                    opportunity = ArbitrageOpportunity(
                        type="cross_platform",
                        event_name=event_name,
                        markets=involved_markets,
                        legs=[
                            {
                                "action": "buy",
                                "outcome": "YES",
                                "price": best_yes_price,
                                "platform": best_yes_market.get("platform"),
                                "market_id": best_yes_market.get("id"),
                                "stake_ratio": best_yes_price / total_cost,
                            },
                            {
                                "action": "buy",
                                "outcome": "NO",
                                "price": best_no_price,
                                "platform": best_no_market.get("platform"),
                                "market_id": best_no_market.get("id"),
                                "stake_ratio": best_no_price / total_cost,
                            },
                        ],
                        margin=margin,
                        margin_percent=margin * 100,
                        total_stake_ratio=total_cost,
                        expected_roi=margin / total_cost,
                        risk_score=risk_score,
                        liquidity_score=liquidity_score,
                        execution_complexity="moderate",
                        estimated_slippage=est_slippage,
                        post_fee_margin=post_fee_margin,
                    )
                    opportunities.append(opportunity)

    # Sort by post-fee margin
    opportunities.sort(key=lambda x: x.post_fee_margin, reverse=True)

    logger.info(f"Found {len(opportunities)} cross-platform arbitrage opportunities")
    return opportunities


# ==================== Single-Platform Mispricing ==================== #

def find_single_platform_mispricing(
    markets: List[Dict[str, Any]],
    threshold: float = 0.05,
    min_liquidity: float = 1000.0,
) -> List[ArbitrageOpportunity]:
    """Find single-platform mispricing where YES + NO ≠ 1.0.

    Parameters
    ----------
    markets : List[Dict[str, Any]]
        List of markets to analyze
    threshold : float
        Minimum deviation from 1.0 required. Default: 5%
    min_liquidity : float
        Minimum liquidity required. Default: $1000

    Returns
    -------
    List[ArbitrageOpportunity]
        List of mispricing opportunities sorted by deviation
    """
    opportunities = []

    for market in markets:
        # Filter by liquidity
        liquidity = market.get("liquidity", 0)
        if liquidity < min_liquidity:
            continue

        prices = market.get("current_prices") or {}
        yes_price = _extract_price(prices, "yes")
        no_price = _extract_price(prices, "no")

        if yes_price is None or no_price is None:
            continue

        # Calculate deviation
        price_sum = yes_price + no_price
        deviation = abs(1.0 - price_sum)

        if deviation > threshold:
            # Determine strategy
            if price_sum < 1.0:
                # Both underpriced - buy both
                strategy = "buy_both"
                margin = 1.0 - price_sum
            else:
                # Both overpriced - sell both (if possible)
                strategy = "sell_both"
                margin = price_sum - 1.0

            liquidity_score = min(liquidity / 50000, 1.0)
            risk_score = _calculate_risk_score("single_platform", margin, liquidity_score)

            avg_fee = _estimate_fees([market])
            est_slippage = _estimate_slippage(liquidity, 1000)
            post_fee_margin = margin - avg_fee - est_slippage

            if post_fee_margin > 0:
                opportunity = ArbitrageOpportunity(
                    type="single_platform",
                    event_name=market.get("title") or market.get("question"),
                    markets=[market],
                    legs=[
                        {
                            "action": "buy" if strategy == "buy_both" else "sell",
                            "outcome": "YES",
                            "price": yes_price,
                            "platform": market.get("platform"),
                            "market_id": market.get("id"),
                            "stake_ratio": 0.5,
                        },
                        {
                            "action": "buy" if strategy == "buy_both" else "sell",
                            "outcome": "NO",
                            "price": no_price,
                            "platform": market.get("platform"),
                            "market_id": market.get("id"),
                            "stake_ratio": 0.5,
                        },
                    ],
                    margin=margin,
                    margin_percent=margin * 100,
                    total_stake_ratio=price_sum if strategy == "buy_both" else 2.0 - price_sum,
                    expected_roi=margin / price_sum if strategy == "buy_both" else margin / (2.0 - price_sum),
                    risk_score=risk_score,
                    liquidity_score=liquidity_score,
                    execution_complexity="simple",
                    estimated_slippage=est_slippage,
                    post_fee_margin=post_fee_margin,
                )
                opportunities.append(opportunity)

    opportunities.sort(key=lambda x: x.post_fee_margin, reverse=True)

    logger.info(f"Found {len(opportunities)} single-platform mispricing opportunities")
    return opportunities


# ==================== Kelly Sizing ==================== #

def calculate_kelly_sizing(
    opportunity: ArbitrageOpportunity,
    bankroll: float,
    fractional_kelly: float = 0.25,
    max_stake_percent: float = 0.10,
) -> Tuple[float, float]:
    """Calculate Kelly-optimal stake size for an arbitrage opportunity.

    Parameters
    ----------
    opportunity : ArbitrageOpportunity
        The arbitrage opportunity
    bankroll : float
        Total bankroll
    fractional_kelly : float
        Fraction of Kelly to use (0.25 = quarter Kelly). Default: 0.25
    max_stake_percent : float
        Maximum stake as percentage of bankroll. Default: 10%

    Returns
    -------
    Tuple[float, float]
        (total_stake, kelly_fraction) - Total stake amount and Kelly fraction used
    """
    # For arbitrage, Kelly fraction is simply the opportunity size
    # relative to the bankroll and the post-fee margin

    post_fee_margin = opportunity.post_fee_margin
    total_stake_ratio = opportunity.total_stake_ratio

    if post_fee_margin <= 0:
        return (0.0, 0.0)

    # Kelly formula for arbitrage: f = margin / (odds - 1)
    # For simple arbitrage with guaranteed payout of 1, simplified to:
    kelly_fraction = post_fee_margin

    # Apply fractional Kelly
    adjusted_kelly = kelly_fraction * fractional_kelly

    # Cap at max stake percent
    adjusted_kelly = min(adjusted_kelly, max_stake_percent)

    # Calculate total stake
    total_stake = bankroll * adjusted_kelly

    logger.debug(f"Kelly sizing: fraction={adjusted_kelly:.4f}, stake=${total_stake:,.2f}")

    return (total_stake, adjusted_kelly)


# ==================== Expected Value ==================== #

def calculate_expected_value(opportunity: ArbitrageOpportunity, stake: float) -> Dict[str, float]:
    """Calculate expected value metrics for an opportunity.

    Parameters
    ----------
    opportunity : ArbitrageOpportunity
        The arbitrage opportunity
    stake : float
        Total stake amount

    Returns
    -------
    Dict[str, float]
        Dictionary with EV, profit, ROI metrics
    """
    post_fee_margin = opportunity.post_fee_margin

    # Guaranteed profit for arbitrage
    expected_profit = stake * post_fee_margin
    roi = post_fee_margin

    # Adjust for risk
    risk_adjusted_ev = expected_profit * (1 - opportunity.risk_score)

    return {
        "expected_profit": expected_profit,
        "roi": roi,
        "roi_percent": roi * 100,
        "risk_adjusted_ev": risk_adjusted_ev,
        "stake": stake,
    }


# ==================== Opportunity Ranking ==================== #

def rank_opportunities(
    opportunities: List[ArbitrageOpportunity],
    bankroll: float,
    weights: Optional[Dict[str, float]] = None,
) -> List[ArbitrageOpportunity]:
    """Rank arbitrage opportunities by composite score.

    Parameters
    ----------
    opportunities : List[ArbitrageOpportunity]
        List of opportunities to rank
    bankroll : float
        Total bankroll for Kelly sizing
    weights : Optional[Dict[str, float]]
        Scoring weights for different factors

    Returns
    -------
    List[ArbitrageOpportunity]
        Sorted list of opportunities with recommended stakes
    """
    if weights is None:
        weights = {
            "margin": 0.40,  # Post-fee margin
            "liquidity": 0.25,  # Liquidity score
            "risk": 0.20,  # Risk score (inverted)
            "complexity": 0.15,  # Execution complexity (inverted)
        }

    complexity_scores = {
        "simple": 1.0,
        "moderate": 0.7,
        "complex": 0.4,
    }

    scored_opportunities = []

    for opp in opportunities:
        # Calculate Kelly sizing
        total_stake, kelly_fraction = calculate_kelly_sizing(opp, bankroll)
        opp.recommended_stake = total_stake
        opp.kelly_fraction = kelly_fraction

        # Calculate composite score
        score = (
            weights["margin"] * opp.post_fee_margin * 10 +  # Scale margin
            weights["liquidity"] * opp.liquidity_score +
            weights["risk"] * (1 - opp.risk_score) +  # Invert risk
            weights["complexity"] * complexity_scores.get(opp.execution_complexity, 0.5)
        )

        scored_opportunities.append((score, opp))

    # Sort by score descending
    scored_opportunities.sort(key=lambda x: x[0], reverse=True)

    ranked_opportunities = [opp for score, opp in scored_opportunities]

    logger.info(f"Ranked {len(ranked_opportunities)} opportunities")
    return ranked_opportunities


# ==================== Multi-Leg Arbitrage (Future Enhancement) ==================== #

def find_multi_leg_arbitrage(
    markets: List[Dict[str, Any]],
    threshold: float = 0.01,
    min_liquidity: float = 1000.0,
) -> List[ArbitrageOpportunity]:
    """Find multi-leg arbitrage in 3+ outcome markets.

    NOTE: This is a placeholder for future implementation.
    Multi-leg arbitrage requires solving a more complex optimization problem.

    Parameters
    ----------
    markets : List[Dict[str, Any]]
        List of markets to analyze
    threshold : float
        Minimum arbitrage margin
    min_liquidity : float
        Minimum liquidity required

    Returns
    -------
    List[ArbitrageOpportunity]
        List of multi-leg arbitrage opportunities
    """
    logger.warning("Multi-leg arbitrage detection not yet implemented")
    return []


# ==================== Threshold Arbitrage (Future Enhancement) ==================== #

def find_threshold_arbitrage(
    markets: List[Dict[str, Any]],
    threshold: float = 0.02,
    min_liquidity: float = 1000.0,
) -> List[ArbitrageOpportunity]:
    """Find threshold market arbitrage (e.g., >50% vs >60%).

    NOTE: This is a placeholder for future implementation.
    Requires finding related threshold markets and analyzing their relationships.

    Parameters
    ----------
    markets : List[Dict[str, Any]]
        List of markets to analyze
    threshold : float
        Minimum arbitrage margin
    min_liquidity : float
        Minimum liquidity required

    Returns
    -------
    List[ArbitrageOpportunity]
        List of threshold arbitrage opportunities
    """
    logger.warning("Threshold arbitrage detection not yet implemented")
    return []
