"""
polyrouter_analysis.py

Market analysis and recommendation engine for prediction markets.

This module provides tools for:
- Market quality scoring (liquidity, spread, volume, trader activity)
- Provider comparison and benchmarking
- Risk-adjusted opportunity ranking
- Historical performance correlation
- Market efficiency analysis

Public API:
- score_market_quality(market)
- compare_providers(markets)
- analyze_market_efficiency(market)
- generate_recommendations(opportunities)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics

__all__ = [
    "MarketQualityScore",
    "ProviderComparison",
    "score_market_quality",
    "score_markets_batch",
    "compare_providers",
    "analyze_market_efficiency",
    "generate_recommendations",
    "calculate_market_health",
]

logger = logging.getLogger(__name__)


@dataclass
class MarketQualityScore:
    """Comprehensive market quality assessment."""

    market_id: str
    platform: str
    title: str

    # Raw metrics
    liquidity: float
    volume_24h: float
    spread_bps: float
    unique_traders: int

    # Normalized scores (0-1)
    liquidity_score: float
    volume_score: float
    spread_score: float
    trader_score: float

    # Composite scores
    overall_score: float
    tradability_score: float  # How easy to trade
    discovery_score: float  # How well price reflects information

    # Risk flags
    risk_flags: List[str]
    warnings: List[str]

    # Recommendations
    recommended_for_arbitrage: bool
    recommended_for_edge_betting: bool
    max_position_size_usd: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_id": self.market_id,
            "platform": self.platform,
            "title": self.title,
            "liquidity": self.liquidity,
            "volume_24h": self.volume_24h,
            "spread_bps": self.spread_bps,
            "unique_traders": self.unique_traders,
            "scores": {
                "liquidity": self.liquidity_score,
                "volume": self.volume_score,
                "spread": self.spread_score,
                "trader": self.trader_score,
                "overall": self.overall_score,
                "tradability": self.tradability_score,
                "discovery": self.discovery_score,
            },
            "risk_flags": self.risk_flags,
            "warnings": self.warnings,
            "recommendations": {
                "arbitrage": self.recommended_for_arbitrage,
                "edge_betting": self.recommended_for_edge_betting,
                "max_position_size": self.max_position_size_usd,
            },
        }


@dataclass
class ProviderComparison:
    """Comparison of multiple providers."""

    providers: List[str]
    market_count_by_provider: Dict[str, int]
    avg_liquidity_by_provider: Dict[str, float]
    avg_volume_by_provider: Dict[str, float]
    avg_spread_by_provider: Dict[str, float]
    best_provider_by_metric: Dict[str, str]
    provider_rankings: List[Tuple[str, float]]  # (provider, score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "providers": self.providers,
            "market_count": self.market_count_by_provider,
            "avg_liquidity": self.avg_liquidity_by_provider,
            "avg_volume": self.avg_volume_by_provider,
            "avg_spread": self.avg_spread_by_provider,
            "best_by_metric": self.best_provider_by_metric,
            "rankings": [
                {"provider": p, "score": s} for p, s in self.provider_rankings
            ],
        }


# ==================== Market Quality Scoring ==================== #

def score_market_quality(
    market: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> MarketQualityScore:
    """Score a single market's quality across multiple dimensions.

    Parameters
    ----------
    market : Dict[str, Any]
        Market data dict
    weights : Optional[Dict[str, float]]
        Scoring weights for different metrics

    Returns
    -------
    MarketQualityScore
        Comprehensive quality assessment
    """
    if weights is None:
        weights = {
            "liquidity": 0.35,
            "volume": 0.30,
            "spread": 0.25,
            "traders": 0.10,
        }

    # Extract raw metrics
    market_id = market.get("id", "")
    platform = market.get("platform", "unknown")
    title = market.get("title") or market.get("question", "")

    liquidity = float(market.get("liquidity") or 0)
    volume_24h = float(market.get("volume_total") or market.get("volume") or 0)
    spread_bps = float(market.get("spread_bps") or 0)
    unique_traders = int(market.get("unique_traders") or 0)

    # Calculate normalized scores (0-1)
    liquidity_score = _score_liquidity(liquidity)
    volume_score = _score_volume(volume_24h)
    spread_score = _score_spread(spread_bps)
    trader_score = _score_traders(unique_traders)

    # Calculate composite scores
    overall_score = (
        weights["liquidity"] * liquidity_score +
        weights["volume"] * volume_score +
        weights["spread"] * spread_score +
        weights["traders"] * trader_score
    )

    # Tradability: liquidity + spread
    tradability_score = (liquidity_score * 0.6 + spread_score * 0.4)

    # Price discovery: volume + traders
    discovery_score = (volume_score * 0.7 + trader_score * 0.3)

    # Identify risk flags
    risk_flags = []
    warnings = []

    if liquidity < 1000:
        risk_flags.append("low_liquidity")
        warnings.append("Liquidity below $1,000 - high slippage risk")

    if volume_24h < 5000:
        risk_flags.append("low_volume")
        warnings.append("24h volume below $5,000 - low market activity")

    if spread_bps > 500:
        risk_flags.append("wide_spread")
        warnings.append("Spread over 500 bps (5%) - poor execution")

    if unique_traders < 10:
        risk_flags.append("few_traders")
        warnings.append("Less than 10 traders - thin market")

    # Generate recommendations
    recommended_for_arbitrage = (
        liquidity_score >= 0.5 and
        spread_score >= 0.6 and
        overall_score >= 0.5
    )

    recommended_for_edge_betting = (
        volume_score >= 0.4 and
        trader_score >= 0.3 and
        overall_score >= 0.4
    )

    # Calculate max safe position size
    max_position_size_usd = min(
        liquidity * 0.05,  # Max 5% of liquidity
        volume_24h * 0.02,  # Max 2% of 24h volume
        10000.0,  # Hard cap at $10k
    )

    return MarketQualityScore(
        market_id=market_id,
        platform=platform,
        title=title,
        liquidity=liquidity,
        volume_24h=volume_24h,
        spread_bps=spread_bps,
        unique_traders=unique_traders,
        liquidity_score=liquidity_score,
        volume_score=volume_score,
        spread_score=spread_score,
        trader_score=trader_score,
        overall_score=overall_score,
        tradability_score=tradability_score,
        discovery_score=discovery_score,
        risk_flags=risk_flags,
        warnings=warnings,
        recommended_for_arbitrage=recommended_for_arbitrage,
        recommended_for_edge_betting=recommended_for_edge_betting,
        max_position_size_usd=max_position_size_usd,
    )


def score_markets_batch(
    markets: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> List[MarketQualityScore]:
    """Score multiple markets in batch.

    Parameters
    ----------
    markets : List[Dict[str, Any]]
        List of market data dicts
    weights : Optional[Dict[str, float]]
        Scoring weights

    Returns
    -------
    List[MarketQualityScore]
        List of quality scores sorted by overall score
    """
    scores = [score_market_quality(m, weights) for m in markets]
    scores.sort(key=lambda s: s.overall_score, reverse=True)
    return scores


# ==================== Scoring Helper Functions ==================== #

def _score_liquidity(liquidity: float) -> float:
    """Score liquidity on 0-1 scale."""
    # Thresholds: $1k = 0.2, $5k = 0.4, $10k = 0.6, $50k = 0.9, $100k+ = 1.0
    if liquidity >= 100000:
        return 1.0
    elif liquidity >= 50000:
        return 0.9
    elif liquidity >= 10000:
        return 0.6
    elif liquidity >= 5000:
        return 0.4
    elif liquidity >= 1000:
        return 0.2
    else:
        return max(liquidity / 1000, 0.0)


def _score_volume(volume: float) -> float:
    """Score 24h volume on 0-1 scale."""
    # Thresholds: $5k = 0.2, $20k = 0.4, $50k = 0.6, $200k = 0.9, $500k+ = 1.0
    if volume >= 500000:
        return 1.0
    elif volume >= 200000:
        return 0.9
    elif volume >= 50000:
        return 0.6
    elif volume >= 20000:
        return 0.4
    elif volume >= 5000:
        return 0.2
    else:
        return max(volume / 5000, 0.0)


def _score_spread(spread_bps: float) -> float:
    """Score spread on 0-1 scale (lower is better)."""
    # Thresholds: <50 bps = 1.0, 100 bps = 0.8, 200 bps = 0.6, 500 bps = 0.2
    if spread_bps <= 50:
        return 1.0
    elif spread_bps <= 100:
        return 0.8
    elif spread_bps <= 200:
        return 0.6
    elif spread_bps <= 500:
        return 0.2
    else:
        return max(1.0 - (spread_bps / 1000), 0.0)


def _score_traders(traders: int) -> float:
    """Score number of traders on 0-1 scale."""
    # Thresholds: 10 = 0.2, 50 = 0.5, 100 = 0.7, 500 = 0.9, 1000+ = 1.0
    if traders >= 1000:
        return 1.0
    elif traders >= 500:
        return 0.9
    elif traders >= 100:
        return 0.7
    elif traders >= 50:
        return 0.5
    elif traders >= 10:
        return 0.2
    else:
        return max(traders / 10, 0.0)


# ==================== Provider Comparison ==================== #

def compare_providers(markets: List[Dict[str, Any]]) -> ProviderComparison:
    """Compare markets across different providers.

    Parameters
    ----------
    markets : List[Dict[str, Any]]
        List of markets from multiple providers

    Returns
    -------
    ProviderComparison
        Comprehensive provider comparison
    """
    # Group markets by provider
    provider_markets = defaultdict(list)
    for market in markets:
        provider = market.get("platform") or market.get("provider", "unknown")
        provider_markets[provider].append(market)

    providers = list(provider_markets.keys())

    # Calculate metrics by provider
    market_count_by_provider = {
        p: len(markets) for p, markets in provider_markets.items()
    }

    avg_liquidity_by_provider = {}
    avg_volume_by_provider = {}
    avg_spread_by_provider = {}

    for provider, provider_mkts in provider_markets.items():
        liquidities = [m.get("liquidity", 0) for m in provider_mkts]
        volumes = [m.get("volume_total") or m.get("volume", 0) for m in provider_mkts]
        spreads = [m.get("spread_bps", 0) for m in provider_mkts]

        avg_liquidity_by_provider[provider] = statistics.mean(liquidities) if liquidities else 0
        avg_volume_by_provider[provider] = statistics.mean(volumes) if volumes else 0
        avg_spread_by_provider[provider] = statistics.mean(spreads) if spreads else 0

    # Find best provider by each metric
    best_provider_by_metric = {}

    if avg_liquidity_by_provider:
        best_provider_by_metric["liquidity"] = max(
            avg_liquidity_by_provider.items(), key=lambda x: x[1]
        )[0]

    if avg_volume_by_provider:
        best_provider_by_metric["volume"] = max(
            avg_volume_by_provider.items(), key=lambda x: x[1]
        )[0]

    if avg_spread_by_provider:
        best_provider_by_metric["spread"] = min(
            avg_spread_by_provider.items(), key=lambda x: x[1]
        )[0]

    # Calculate overall provider rankings
    provider_scores = {}
    for provider in providers:
        liquidity = avg_liquidity_by_provider.get(provider, 0)
        volume = avg_volume_by_provider.get(provider, 0)
        spread = avg_spread_by_provider.get(provider, 0)

        # Calculate composite score
        liquidity_score = _score_liquidity(liquidity)
        volume_score = _score_volume(volume)
        spread_score = _score_spread(spread)

        composite_score = (
            liquidity_score * 0.4 +
            volume_score * 0.4 +
            spread_score * 0.2
        )

        provider_scores[provider] = composite_score

    # Sort providers by score
    provider_rankings = sorted(
        provider_scores.items(), key=lambda x: x[1], reverse=True
    )

    logger.info(f"Compared {len(providers)} providers across {len(markets)} markets")

    return ProviderComparison(
        providers=providers,
        market_count_by_provider=market_count_by_provider,
        avg_liquidity_by_provider=avg_liquidity_by_provider,
        avg_volume_by_provider=avg_volume_by_provider,
        avg_spread_by_provider=avg_spread_by_provider,
        best_provider_by_metric=best_provider_by_metric,
        provider_rankings=provider_rankings,
    )


# ==================== Market Efficiency Analysis ==================== #

def analyze_market_efficiency(market: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze market efficiency and price discovery quality.

    Parameters
    ----------
    market : Dict[str, Any]
        Market data dict

    Returns
    -------
    Dict[str, Any]
        Efficiency analysis with scores and insights
    """
    prices = market.get("current_prices") or {}

    # Extract YES/NO prices
    yes_price = None
    no_price = None

    if "yes" in prices:
        yes_val = prices["yes"]
        yes_price = yes_val.get("price") if isinstance(yes_val, dict) else float(yes_val)
    if "no" in prices:
        no_val = prices["no"]
        no_price = no_val.get("price") if isinstance(no_val, dict) else float(no_val)

    if yes_price is None or no_price is None:
        return {
            "efficiency_score": 0.0,
            "insights": ["Insufficient price data"],
        }

    # Calculate price sum
    price_sum = yes_price + no_price

    # Efficiency metrics
    price_sum_deviation = abs(1.0 - price_sum)

    # Spread analysis
    spread = abs(yes_price - no_price)
    spread_bps = market.get("spread_bps", 0)

    # Volume analysis
    volume = market.get("volume_total") or market.get("volume", 0)
    liquidity = market.get("liquidity", 0)

    # Calculate efficiency score (0-1, higher is better)
    # Efficient markets have: price_sum ≈ 1.0, low spreads, high volume
    sum_efficiency = 1.0 - min(price_sum_deviation / 0.10, 1.0)  # Normalize to 10% deviation
    spread_efficiency = _score_spread(spread_bps)
    volume_efficiency = _score_volume(volume)

    efficiency_score = (
        sum_efficiency * 0.5 +
        spread_efficiency * 0.3 +
        volume_efficiency * 0.2
    )

    # Generate insights
    insights = []

    if price_sum_deviation > 0.05:
        insights.append(f"Price sum deviates by {price_sum_deviation:.1%} from 1.0 - potential arbitrage")
    else:
        insights.append("Price sum near 1.0 - well-calibrated market")

    if spread_bps > 200:
        insights.append(f"Wide spread ({spread_bps} bps) - illiquid market")
    elif spread_bps < 50:
        insights.append(f"Tight spread ({spread_bps} bps) - liquid market")

    if volume < 10000:
        insights.append("Low trading volume - price discovery may be weak")
    elif volume > 100000:
        insights.append("High trading volume - strong price discovery")

    if efficiency_score >= 0.8:
        recommendation = "Highly efficient market - good for edge betting"
    elif efficiency_score >= 0.6:
        recommendation = "Moderately efficient market - suitable for trading"
    elif efficiency_score >= 0.4:
        recommendation = "Somewhat inefficient market - check for arbitrage"
    else:
        recommendation = "Inefficient market - high risk, potential opportunities"

    return {
        "efficiency_score": efficiency_score,
        "price_sum": price_sum,
        "price_sum_deviation": price_sum_deviation,
        "spread_bps": spread_bps,
        "volume": volume,
        "liquidity": liquidity,
        "insights": insights,
        "recommendation": recommendation,
    }


# ==================== Market Health ==================== #

def calculate_market_health(market: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive market health metrics.

    Parameters
    ----------
    market : Dict[str, Any]
        Market data dict

    Returns
    -------
    Dict[str, Any]
        Health metrics and status
    """
    quality_score = score_market_quality(market)
    efficiency = analyze_market_efficiency(market)

    # Calculate overall health score
    health_score = (
        quality_score.overall_score * 0.6 +
        efficiency["efficiency_score"] * 0.4
    )

    # Determine health status
    if health_score >= 0.8:
        status = "excellent"
        color = "green"
    elif health_score >= 0.6:
        status = "good"
        color = "blue"
    elif health_score >= 0.4:
        status = "fair"
        color = "yellow"
    else:
        status = "poor"
        color = "red"

    return {
        "health_score": health_score,
        "status": status,
        "color": color,
        "quality_score": quality_score.overall_score,
        "efficiency_score": efficiency["efficiency_score"],
        "risk_flags": quality_score.risk_flags,
        "warnings": quality_score.warnings,
        "insights": efficiency["insights"],
        "recommendation": efficiency["recommendation"],
    }


# ==================== Recommendation Engine ==================== #

def generate_recommendations(
    opportunities: List[Any],  # List of ArbitrageOpportunity or similar
    market_scores: List[MarketQualityScore],
    max_recommendations: int = 10,
) -> List[Dict[str, Any]]:
    """Generate ranked recommendations combining opportunities and market quality.

    Parameters
    ----------
    opportunities : List[Any]
        List of trading opportunities
    market_scores : List[MarketQualityScore]
        Market quality assessments
    max_recommendations : int
        Maximum number of recommendations to return

    Returns
    -------
    List[Dict[str, Any]]
        Ranked recommendations with rationale
    """
    recommendations = []

    # Create market quality lookup
    quality_by_id = {score.market_id: score for score in market_scores}

    for opp in opportunities:
        # Get quality scores for involved markets
        involved_market_ids = [m.get("id") for m in opp.markets if m.get("id")]
        quality_scores = [quality_by_id.get(mid) for mid in involved_market_ids]
        quality_scores = [q for q in quality_scores if q is not None]

        if not quality_scores:
            continue

        # Calculate average quality
        avg_quality = statistics.mean(q.overall_score for q in quality_scores)

        # Generate rationale
        rationale = []
        rationale.append(f"{opp.margin_percent:.2f}% margin after fees")

        if opp.post_fee_margin > 0.03:
            rationale.append("Strong margin (>3%)")
        elif opp.post_fee_margin > 0.01:
            rationale.append("Moderate margin (>1%)")

        if opp.liquidity_score > 0.7:
            rationale.append("High liquidity")
        elif opp.liquidity_score > 0.4:
            rationale.append("Adequate liquidity")

        if opp.risk_score < 0.3:
            rationale.append("Low risk")
        elif opp.risk_score < 0.6:
            rationale.append("Moderate risk")
        else:
            rationale.append("Higher risk - careful execution needed")

        # Assign recommendation strength
        if opp.post_fee_margin > 0.03 and opp.liquidity_score > 0.6 and avg_quality > 0.6:
            strength = "strong"
        elif opp.post_fee_margin > 0.01 and opp.liquidity_score > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        recommendations.append({
            "opportunity": opp.to_dict(),
            "quality_scores": [q.to_dict() for q in quality_scores],
            "avg_quality": avg_quality,
            "rationale": rationale,
            "strength": strength,
        })

    # Sort by combination of margin and quality
    recommendations.sort(
        key=lambda r: (
            r["opportunity"]["post_fee_margin"] * 0.7 +
            r["avg_quality"] * 0.3
        ),
        reverse=True
    )

    # Limit to max recommendations
    recommendations = recommendations[:max_recommendations]

    logger.info(f"Generated {len(recommendations)} recommendations")

    return recommendations
