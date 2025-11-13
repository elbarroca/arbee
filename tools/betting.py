"""
Professional Bettor Tools

Provides comprehensive betting analysis tools for market evaluation, position sizing,
risk assessment, and recommendation generation with strict validation.
"""
import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.tools import tool
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import norm

logger = logging.getLogger(__name__)


def _calculate_liquidity_score(liquidity: float) -> float:
    """Calculate liquidity score from 0-100 based on USD liquidity."""
    assert liquidity >= 0, f"Liquidity must be non-negative: {liquidity}"

    if liquidity >= 50000:
        return 100.0
    elif liquidity >= 10000:
        return 70.0 + (liquidity - 10000) / (50000 - 10000) * 30.0
    elif liquidity >= 1000:
        return 40.0 + (liquidity - 1000) / (10000 - 1000) * 30.0
    else:
        return max(0.0, (liquidity / 1000) * 40.0)


def _calculate_spread_metrics(spread: float) -> tuple[str, float]:
    """Calculate spread quality and score."""
    assert 0 <= spread <= 1, f"Spread must be between 0-1: {spread}"

    if spread < 0.01:
        return "excellent", 100.0
    elif spread < 0.02:
        return "good", 80.0
    elif spread < 0.05:
        return "acceptable", 60.0
    else:
        return "wide", 30.0


def _calculate_volume_score(volume: float) -> tuple[str, float]:
    """Calculate volume level and score."""
    assert volume >= 0, f"Volume must be non-negative: {volume}"

    if volume >= 100000:
        return "high", 100.0
    elif volume >= 10000:
        return "medium", 60.0 + (volume - 10000) / (100000 - 10000) * 40.0
    else:
        return "low", max(0.0, (volume / 10000) * 60.0)


def _analyze_orderbook(orderbook: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Analyze orderbook depth and imbalance."""
    if not orderbook:
        return None

    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    mid_price = orderbook.get("mid_price", 0.5)

    assert 0 < mid_price < 1, f"Invalid mid price: {mid_price}"

    # Calculate bid/ask volumes
    bid_volume = sum(float(b.get("size", 0)) for b in bids[:10])
    ask_volume = sum(float(a.get("size", 0)) for a in asks[:10])

    if bid_volume + ask_volume > 0:
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    else:
        imbalance = 0.0

    # Depth at 1% from mid
    depth_1pct = sum(
        float(b.get("size", 0)) for b in bids
        if abs(float(b.get("price", 0)) - mid_price) <= 0.01
    )

    return {
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "imbalance": round(imbalance, 3),
        "depth_1pct": depth_1pct,
        "num_bids": len(bids),
        "num_asks": len(asks)
    }


@tool
def analyze_market_context_tool(
    market_data: Dict[str, Any],
    orderbook: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze market context including liquidity, volume, and orderbook depth.

    Essential for understanding if a market has sufficient liquidity for betting
    and identifying potential execution risks (slippage, market impact).

    Args:
        market_data: Market information with volume, liquidity, spread
        orderbook: Optional orderbook data with bids/asks

    Returns:
        Dict with liquidity_score, spread_analysis, volume_metrics, execution_risk

    Raises:
        AssertionError: If market data contains invalid values
    """
    # Extract and validate market metrics
    volume = float(market_data.get("volume", 0))
    liquidity = float(market_data.get("liquidity", 0))
    spread = float(market_data.get("spread", 0))

    assert volume >= 0, f"Volume must be non-negative: {volume}"
    assert liquidity >= 0, f"Liquidity must be non-negative: {liquidity}"
    assert 0 <= spread <= 1, f"Spread must be between 0-1: {spread}"

    # Calculate metrics using helper functions
    liquidity_score = _calculate_liquidity_score(liquidity)
    spread_quality, spread_score = _calculate_spread_metrics(spread)
    volume_level, volume_score = _calculate_volume_score(volume)
    orderbook_analysis = _analyze_orderbook(orderbook)

    # Calculate execution risk and market quality scores
    execution_risk_score = spread_score * 0.4 + liquidity_score * 0.4 + volume_score * 0.2
    market_quality_score = liquidity_score * 0.5 + spread_score * 0.3 + volume_score * 0.2

    # Determine risk levels
    if execution_risk_score >= 80:
        execution_risk = "low"
    elif execution_risk_score >= 60:
        execution_risk = "medium"
    else:
        execution_risk = "high"

    # Generate recommendation
    if market_quality_score >= 80:
        recommendation = "Market has excellent liquidity and tight spread - suitable for betting"
    elif market_quality_score >= 60:
        recommendation = "Market has acceptable liquidity - proceed with caution"
    else:
        recommendation = "Market liquidity is low - high execution risk, consider smaller size"

    result = {
        "liquidity_score": round(liquidity_score, 1),
        "liquidity_usd": liquidity,
        "spread": spread,
        "spread_bps": int(spread * 10000),
        "spread_quality": spread_quality,
        "spread_score": round(spread_score, 1),
        "volume": volume,
        "volume_level": volume_level,
        "volume_score": round(volume_score, 1),
        "execution_risk": execution_risk,
        "execution_risk_score": round(execution_risk_score, 1),
        "market_quality_score": round(market_quality_score, 1),
        "orderbook_analysis": orderbook_analysis,
        "recommendation": recommendation
    }

    logger.info(
        f"Market analysis: quality={market_quality_score:.1f}, "
        f"liquidity=${liquidity:,.0f}, spread={spread:.2%}"
    )

    return result


def _calculate_kelly_fraction(p_bayesian: float, market_price: float, kelly_fraction: float) -> tuple[float, float]:
    """Calculate full and fractional Kelly bet sizes."""
    assert 0 < market_price < 1, f"Market price must be between 0-1: {market_price}"
    assert 0 <= p_bayesian <= 1, f"Bayesian probability must be between 0-1: {p_bayesian}"
    assert 0 <= kelly_fraction <= 1, f"Kelly fraction must be between 0-1: {kelly_fraction}"

    b = (1 - market_price) / market_price  # decimal odds
    p = p_bayesian
    q = 1 - p_bayesian

    kelly_full = (b * p - q) / b
    kelly_fractional = max(0.0, kelly_full * kelly_fraction)

    return kelly_full, kelly_fractional


def _assess_edge_value(adjusted_edge: float) -> tuple[str, int]:
    """Assess the value quality based on adjusted edge."""
    assert -1 <= adjusted_edge <= 1, f"Adjusted edge must be between -1 and 1: {adjusted_edge}"

    if adjusted_edge >= 0.10:
        return "excellent", 100
    elif adjusted_edge >= 0.05:
        return "good", 80
    elif adjusted_edge >= 0.03:
        return "fair", 60
    elif adjusted_edge >= 0.01:
        return "marginal", 40
    else:
        return "no_value", 0


def _should_bet_decision(adjusted_edge: float, confidence: int, kelly_fractional: float) -> bool:
    """Determine if betting is recommended based on criteria."""
    return (
        adjusted_edge > 0.02 and
        0 <= confidence <= 100 and
        confidence >= 50 and
        kelly_fractional > 0
    )


@tool
def evaluate_betting_edge_tool(
    p_bayesian: float,
    market_price: float,
    confidence: int,
    kelly_fraction: float = 0.25
) -> Dict[str, Any]:
    """
    Evaluate betting edge by comparing Bayesian probability vs market price.

    Determines if there's a profitable opportunity (positive expected value)
    and quantifies the strength of the edge.

    Args:
        p_bayesian: True probability estimate (0-1)
        market_price: Current market price (0-1)
        confidence: Confidence in probability estimate (0-100)
        kelly_fraction: Fraction of Kelly to use (default: 0.25 for 25%)

    Returns:
        Dict with edge, expected_value, value_assessment, should_bet

    Raises:
        AssertionError: If input parameters are invalid
    """
    # Validate inputs
    assert 0 <= p_bayesian <= 1, f"Bayesian probability must be between 0-1: {p_bayesian}"
    assert 0 <= market_price <= 1, f"Market price must be between 0-1: {market_price}"
    assert 0 <= confidence <= 100, f"Confidence must be between 0-100: {confidence}"
    assert 0 <= kelly_fraction <= 1, f"Kelly fraction must be between 0-1: {kelly_fraction}"

    # Calculate edge metrics
    edge = p_bayesian - market_price
    edge_pct = edge * 100
    confidence_factor = confidence / 100
    adjusted_edge = edge * confidence_factor

    # Calculate expected value for binary market
    expected_value = (p_bayesian * (1 - market_price)) - ((1 - p_bayesian) * market_price)
    expected_value_pct = expected_value * 100

    # Calculate Kelly sizing
    kelly_full, kelly_fractional = _calculate_kelly_fraction(p_bayesian, market_price, kelly_fraction)

    # Assess value and make betting decision
    value_assessment, value_score = _assess_edge_value(adjusted_edge)
    should_bet = _should_bet_decision(adjusted_edge, confidence, kelly_fractional)

    # Calculate risk-reward ratio
    risk_reward_ratio = (1 - market_price) / market_price if market_price > 0 else 0.0

    result = {
        "edge": round(edge, 4),
        "edge_pct": round(edge_pct, 2),
        "adjusted_edge": round(adjusted_edge, 4),
        "adjusted_edge_pct": round(adjusted_edge * 100, 2),
        "expected_value": round(expected_value, 4),
        "expected_value_pct": round(expected_value_pct, 2),
        "kelly_full": round(kelly_full, 4),
        "kelly_fractional": round(kelly_fractional, 4),
        "kelly_pct": round(kelly_fractional * 100, 2),
        "value_assessment": value_assessment,
        "value_score": value_score,
        "should_bet": should_bet,
        "risk_reward_ratio": round(risk_reward_ratio, 2),
        "confidence_factor": confidence_factor,
        "reasoning": (
            f"Edge of {edge_pct:.1f}% ({value_assessment}) with {confidence}% confidence. "
            f"EV: {expected_value_pct:.1f}%. "
            f"{'RECOMMEND BET' if should_bet else 'DO NOT BET'}"
        )
    }

    logger.info(
        f"Edge evaluation: {edge_pct:.1f}% edge, "
        f"Kelly={kelly_fractional:.2%}, should_bet={should_bet}"
    )

    return result


def _get_kelly_multiplier(confidence: int) -> float:
    """Determine Kelly fraction based on confidence level."""
    assert 0 <= confidence <= 100, f"Confidence must be between 0-100: {confidence}"

    if confidence >= 80:
        return 0.5
    elif confidence >= 60:
        return 0.25
    else:
        return 0.1


def _calculate_size_category(size_pct: float) -> str:
    """Categorize position size based on percentage of bankroll."""
    assert size_pct >= 0, f"Size percentage must be non-negative: {size_pct}"

    if size_pct >= 4:
        return "large"
    elif size_pct >= 2:
        return "medium"
    elif size_pct >= 0.5:
        return "small"
    else:
        return "micro"


def _build_position_rationale(recommended_size: float, size_pct: float, kelly_adjusted: float,
                             constraints: list) -> str:
    """Generate human-readable rationale for position sizing."""
    rationale = (
        f"Recommended ${recommended_size:,.2f} ({size_pct:.1f}% of bankroll). "
        f"Kelly: {kelly_adjusted:.2%}."
    )

    if constraints:
        rationale += " Constraints: " + ", ".join(constraints) + "."

    return rationale


@tool
def calculate_position_size_tool(
    edge: float,
    bankroll: float,
    confidence: int,
    risk_limits: Dict[str, float],
    market_liquidity: float = 10000.0
) -> Dict[str, Any]:
    """
    Calculate optimal position size using Kelly criterion with risk constraints.

    Uses fractional Kelly (25-50%) for conservative sizing, applies hard limits
    from risk management, and adjusts for market liquidity.

    Args:
        edge: Betting edge (0-1)
        bankroll: Current bankroll (USD)
        confidence: Confidence level (0-100)
        risk_limits: Dict with max_position_size_pct, max_exposure_pct
        market_liquidity: Market liquidity in USD

    Returns:
        Dict with recommended_size, kelly_fraction, size_rationale

    Raises:
        AssertionError: If input parameters are invalid
    """
    # Validate inputs
    assert 0 <= edge <= 1, f"Edge must be between 0-1: {edge}"
    assert bankroll > 0, f"Bankroll must be positive: {bankroll}"
    assert 0 <= confidence <= 100, f"Confidence must be between 0-100: {confidence}"
    assert market_liquidity > 0, f"Market liquidity must be positive: {market_liquidity}"

    # Extract risk limits with defaults
    max_position_pct = risk_limits.get("max_position_size_pct", 0.05)
    assert 0 < max_position_pct <= 1, f"Max position percentage must be between 0-1: {max_position_pct}"

    # Calculate Kelly parameters
    kelly_multiplier = _get_kelly_multiplier(confidence)
    kelly_full = max(0.0, edge)
    kelly_fractional = kelly_full * kelly_multiplier
    confidence_adj = confidence / 100
    kelly_adjusted = kelly_fractional * confidence_adj

    # Calculate position sizes with constraints
    kelly_size = bankroll * kelly_adjusted
    max_size = bankroll * max_position_pct
    liquidity_limit = market_liquidity * 0.10

    # Apply constraints
    recommended_size = min(kelly_size, max_size, liquidity_limit)
    liquidity_constrained = recommended_size == liquidity_limit

    # Round and calculate percentages
    recommended_size = round(recommended_size, 2)
    size_pct = (recommended_size / bankroll * 100) if bankroll > 0 else 0.0

    # Categorize size and build rationale
    size_category = _calculate_size_category(size_pct)

    constraints = []
    if liquidity_constrained:
        constraints.append("limited by market liquidity")
    if kelly_size > max_size:
        constraints.append(f"capped at {max_position_pct * 100}% of bankroll")
    if kelly_multiplier < 0.5:
        constraints.append(f"using {kelly_multiplier}x Kelly for safety")

    rationale = _build_position_rationale(recommended_size, size_pct, kelly_adjusted, constraints)

    result = {
        "recommended_size": recommended_size,
        "recommended_size_pct": round(size_pct, 2),
        "kelly_full": round(kelly_full, 4),
        "kelly_fractional": round(kelly_fractional, 4),
        "kelly_adjusted": round(kelly_adjusted, 4),
        "kelly_multiplier": kelly_multiplier,
        "kelly_size": round(kelly_size, 2),
        "max_size": round(max_size, 2),
        "liquidity_limit": round(liquidity_limit, 2),
        "liquidity_constrained": liquidity_constrained,
        "size_category": size_category,
        "rationale": rationale
    }

    logger.info(
        f"Position size: ${recommended_size:,.2f} ({size_pct:.1f}%), "
        f"Kelly={kelly_adjusted:.2%}"
    )

    return result


def _analyze_category_concentration(current_positions: List[Dict[str, Any]],
                                   proposed_bet: Dict[str, Any],
                                   total_equity: float) -> tuple[Dict[str, float], float]:
    """Analyze category concentration in portfolio."""
    category_exposure = {}

    # Aggregate current positions by category
    for pos in current_positions:
        cat = pos.get("category", "unknown")
        size = float(pos.get("size", 0))
        category_exposure[cat] = category_exposure.get(cat, 0) + size

    # Add proposed bet
    proposed_category = proposed_bet.get("category", "unknown")
    proposed_size = float(proposed_bet.get("size", 0))
    category_exposure[proposed_category] = category_exposure.get(proposed_category, 0) + proposed_size

    # Calculate max concentration
    max_category_exposure = max(category_exposure.values()) if category_exposure else 0
    max_category_pct = (max_category_exposure / total_equity * 100) if total_equity > 0 else 0

    return category_exposure, max_category_pct


def _analyze_provider_concentration(current_positions: List[Dict[str, Any]],
                                   proposed_bet: Dict[str, Any],
                                   total_equity: float) -> tuple[Dict[str, float], float]:
    """Analyze provider concentration in portfolio."""
    provider_exposure = {}

    # Aggregate current positions by provider
    for pos in current_positions:
        prov = pos.get("provider", "unknown")
        size = float(pos.get("size", 0))
        provider_exposure[prov] = provider_exposure.get(prov, 0) + size

    # Add proposed bet
    proposed_provider = proposed_bet.get("provider", "unknown")
    proposed_size = float(proposed_bet.get("size", 0))
    provider_exposure[proposed_provider] = provider_exposure.get(proposed_provider, 0) + proposed_size

    # Calculate max concentration
    max_provider_exposure = max(provider_exposure.values()) if provider_exposure else 0
    max_provider_pct = (max_provider_exposure / total_equity * 100) if total_equity > 0 else 0

    return provider_exposure, max_provider_pct


def _find_correlated_positions(current_positions: List[Dict[str, Any]],
                              proposed_slug: str) -> List[Dict[str, Any]]:
    """Find positions correlated with proposed bet using keyword matching."""
    correlated_positions = []
    proposed_keywords = set(proposed_slug.lower().split('-'))

    for pos in current_positions:
        pos_slug = pos.get("market_slug", "")
        pos_keywords = set(pos_slug.lower().split('-'))

        # Calculate keyword overlap
        overlap = len(proposed_keywords & pos_keywords)
        if overlap >= 2:  # At least 2 common keywords
            correlated_positions.append({
                "market_slug": pos_slug,
                "overlap_score": overlap / max(len(proposed_keywords), len(pos_keywords)),
                "category": pos.get("category")
            })

    return correlated_positions


def _assess_correlation_risk(correlated_positions: List[Dict[str, Any]]) -> tuple[str, float]:
    """Assess correlation risk based on number of correlated positions."""
    num_correlated = len(correlated_positions)

    if num_correlated >= 3:
        return "high", 30.0
    elif num_correlated >= 1:
        return "medium", 60.0
    else:
        return "low", 90.0


def _assess_concentration_risk(max_category_pct: float, max_provider_pct: float) -> tuple[str, float]:
    """Assess concentration risk based on max exposures."""
    if max_category_pct > 40 or max_provider_pct > 50:
        return "high", 30.0
    elif max_category_pct > 25 or max_provider_pct > 35:
        return "medium", 60.0
    else:
        return "low", 90.0


def _calculate_diversification_score(num_positions: int, num_categories: int,
                                    correlation_score: float) -> float:
    """Calculate overall diversification score."""
    return (
        min(100.0, (num_positions / 10) * 30) +  # More positions = better (up to 10)
        min(100.0, (num_categories / 5) * 40) +  # More categories = better (up to 5)
        correlation_score * 0.3                   # Less correlation = better
    )


def _generate_risk_warnings(max_category_pct: float, proposed_category: str,
                           correlated_positions: List[Dict[str, Any]],
                           num_positions: int) -> List[str]:
    """Generate risk warnings based on portfolio analysis."""
    warnings = []

    if max_category_pct > 30:
        warnings.append(
            f"High concentration in '{proposed_category}' category ({max_category_pct:.1f}% of portfolio)"
        )
    if len(correlated_positions) >= 2:
        warnings.append(
            f"Proposed bet correlates with {len(correlated_positions)} existing positions"
        )
    if num_positions + 1 > 15:
        warnings.append(f"Portfolio has {num_positions + 1} positions - may be over-diversified")

    return warnings


@tool
def assess_portfolio_risk_tool(
    current_positions: List[Dict[str, Any]],
    proposed_bet: Dict[str, Any],
    portfolio_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess portfolio risk including correlation, concentration, and diversification.

    Analyzes how proposed bet affects overall portfolio risk, checks for
    over-concentration in specific categories/providers, and identifies
    correlated positions.

    Args:
        current_positions: List of open positions with category, provider, size
        proposed_bet: Proposed bet with market_slug, category, provider, size
        portfolio_metrics: Current portfolio metrics (exposure, num_positions, etc.)

    Returns:
        Dict with correlation_risk, concentration_risk, diversification_score, warnings

    Raises:
        AssertionError: If portfolio data contains invalid values
    """
    # Extract proposed bet details
    proposed_category = proposed_bet.get("category", "unknown")
    proposed_provider = proposed_bet.get("provider", "unknown")
    proposed_size = float(proposed_bet.get("size", 0))
    proposed_slug = proposed_bet.get("market_slug", "")

    assert proposed_size >= 0, f"Proposed bet size must be non-negative: {proposed_size}"

    # Current portfolio state
    total_exposure = float(portfolio_metrics.get("total_exposure", 0))
    num_positions = int(portfolio_metrics.get("num_positions", 0))
    total_equity = float(portfolio_metrics.get("total_equity", 10000))

    assert total_equity > 0, f"Total equity must be positive: {total_equity}"

    # Analyze concentrations
    category_exposure, max_category_pct = _analyze_category_concentration(
        current_positions, proposed_bet, total_equity
    )
    provider_exposure, max_provider_pct = _analyze_provider_concentration(
        current_positions, proposed_bet, total_equity
    )

    # Find correlated positions
    correlated_positions = _find_correlated_positions(current_positions, proposed_slug)

    # Assess risks
    correlation_risk, correlation_score = _assess_correlation_risk(correlated_positions)
    concentration_risk, concentration_score = _assess_concentration_risk(max_category_pct, max_provider_pct)

    # Calculate diversification
    num_categories = len(set(cat for cat in category_exposure.keys() if cat != "unknown"))
    num_providers = len(set(prov for prov in provider_exposure.keys() if prov != "unknown"))

    diversification_score = _calculate_diversification_score(num_positions, num_categories, correlation_score)

    # Overall portfolio risk
    portfolio_risk_score = (
        concentration_score * 0.4 +
        correlation_score * 0.3 +
        diversification_score * 0.3
    )

    if portfolio_risk_score >= 75:
        overall_risk = "low"
    elif portfolio_risk_score >= 50:
        overall_risk = "medium"
    else:
        overall_risk = "high"

    # Generate warnings and recommendation
    warnings = _generate_risk_warnings(max_category_pct, proposed_category, correlated_positions, num_positions)

    if overall_risk == "low":
        recommendation = "Portfolio risk is well-managed - safe to proceed"
    elif overall_risk == "medium":
        recommendation = "Moderate portfolio risk - consider position correlation"
    else:
        recommendation = "High portfolio risk - diversify or reduce exposure"

    # Find max category name
    max_category = "none"
    if category_exposure:
        max_category_exposure = max(category_exposure.values())
        max_category = list(category_exposure.keys())[list(category_exposure.values()).index(max_category_exposure)]

    result = {
        "correlation_risk": correlation_risk,
        "correlation_score": round(correlation_score, 1),
        "correlated_positions": correlated_positions[:3],  # Top 3
        "num_correlated": len(correlated_positions),
        "concentration_risk": concentration_risk,
        "concentration_score": round(concentration_score, 1),
        "max_category": max_category,
        "max_category_pct": round(max_category_pct, 1),
        "max_provider_pct": round(max_provider_pct, 1),
        "diversification_score": round(diversification_score, 1),
        "num_categories": num_categories,
        "num_providers": num_providers,
        "overall_risk": overall_risk,
        "portfolio_risk_score": round(portfolio_risk_score, 1),
        "warnings": warnings,
        "recommendation": recommendation
    }

    logger.info(
        f"Portfolio risk: {overall_risk} (score={portfolio_risk_score:.1f}), "
        f"correlation={len(correlated_positions)}, concentration={max_category_pct:.1f}%"
    )

    return result


def _extract_rationale_metrics(edge_analysis: Dict[str, Any], market_analysis: Dict[str, Any],
                              portfolio_analysis: Dict[str, Any], position_size: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from analysis results for rationale generation."""
    return {
        "edge_pct": edge_analysis.get("edge_pct", 0),
        "should_bet": edge_analysis.get("should_bet", False),
        "ev_pct": edge_analysis.get("expected_value_pct", 0),
        "kelly_pct": edge_analysis.get("kelly_pct", 0),
        "market_quality": market_analysis.get("market_quality_score", 0),
        "liquidity": market_analysis.get("liquidity_usd", 0),
        "execution_risk": market_analysis.get("execution_risk", "unknown"),
        "portfolio_risk": portfolio_analysis.get("overall_risk", "unknown"),
        "warnings": portfolio_analysis.get("warnings", []),
        "rec_size": position_size.get("recommended_size", 0),
        "size_pct": position_size.get("recommended_size_pct", 0)
    }


def _build_market_assessment_section(market_question: str, p_bayesian: float,
                                    market_price: float, edge_pct: float,
                                    ev_pct: float, value_assessment: str) -> List[str]:
    """Build market assessment section."""
    return [
        "## MARKET ASSESSMENT",
        f"**Question**: {market_question}",
        f"**True Probability (Bayesian)**: {p_bayesian:.1%}",
        f"**Market Price**: {market_price:.1%}",
        f"**Edge**: {edge_pct:+.1f}% ({value_assessment})",
        f"**Expected Value**: {ev_pct:+.1f}%",
        ""
    ]


def _build_evidence_section(top_evidence: List[Dict[str, Any]]) -> List[str]:
    """Build key evidence section if evidence is provided."""
    if not top_evidence:
        return []

    sections = ["## KEY SUPPORTING EVIDENCE"]
    for i, evidence in enumerate(top_evidence[:5], 1):
        title = evidence.get("title", "")
        support = evidence.get("support", "")
        llr = evidence.get("estimated_LLR", 0)
        sections.append(f"{i}. {title} [{support}, LLR={llr:+.2f}]")
    sections.append("")
    return sections


def _build_liquidity_section(market_quality: float, liquidity: float,
                            execution_risk: str, spread: float,
                            spread_quality: str) -> List[str]:
    """Build market liquidity and execution section."""
    return [
        "## MARKET LIQUIDITY & EXECUTION",
        f"**Liquidity**: ${liquidity:,.0f} (score: {market_quality:.0f}/100)",
        f"**Execution Risk**: {execution_risk}",
        f"**Spread**: {spread:.1%} ({spread_quality})",
        ""
    ]


def _build_portfolio_section(portfolio_risk: str, correlation_risk: str,
                            concentration_risk: str, warnings: List[str]) -> List[str]:
    """Build portfolio impact section."""
    sections = [
        "## PORTFOLIO IMPACT",
        f"**Portfolio Risk**: {portfolio_risk}",
        f"**Correlation Risk**: {correlation_risk}",
        f"**Concentration Risk**: {concentration_risk}"
    ]

    if warnings:
        sections.append(f"**Warnings**: {'; '.join(warnings)}")
    sections.append("")
    return sections


def _build_position_section(should_bet: bool, rec_size: float, size_pct: float,
                           kelly_pct: float, size_category: str,
                           rationale: str, reasoning: str) -> List[str]:
    """Build recommended position section."""
    sections = ["## RECOMMENDED POSITION"]

    if should_bet:
        sections.extend([
            f"**Size**: ${rec_size:,.2f} ({size_pct:.1f}% of bankroll)",
            f"**Kelly Fraction**: {kelly_pct:.1f}%",
            f"**Category**: {size_category}",
            f"**Rationale**: {rationale}"
        ])
    else:
        sections.extend([
            "**Recommendation**: DO NOT BET",
            f"**Reason**: {reasoning}"
        ])

    sections.append("")
    return sections


def _identify_key_risks(execution_risk: str, portfolio_risk: str,
                       edge_pct: float, market_price: float) -> List[str]:
    """Identify and describe key risks."""
    risks = []

    if execution_risk in ("medium", "high"):
        risks.append(f"Execution risk is {execution_risk} - possible slippage on entry/exit")

    if portfolio_risk in ("medium", "high"):
        risks.append(f"Portfolio risk is {portfolio_risk} - diversification concerns")

    if edge_pct < 5:
        risks.append("Edge is relatively small - sensitive to probability estimation errors")

    if market_price > 0.8 or market_price < 0.2:
        risks.append("Market price is extreme - binary outcomes have high variance")

    if not risks:
        risks.append("No major risks identified with this bet")

    return [f"- {risk}" for risk in risks]


def _build_alternatives_section(edge_pct: float, rec_size: float, market_price: float) -> List[str]:
    """Build alternatives considered section."""
    sections = ["## ALTERNATIVES CONSIDERED"]

    # Alternative 1: Don't bet
    sections.append("1. **Don't Bet**: Preserve capital if edge < 3% or confidence < 60%")

    # Alternative 2: Smaller size
    if rec_size > 100:
        sections.append(f"2. **Smaller Size**: Start with 50% of recommended (${rec_size * 0.5:,.2f})")

    # Alternative 3: Wait for better price
    if abs(edge_pct) < 5:
        better_price = market_price - 0.05 if edge_pct > 0 else market_price + 0.05
        sections.append(f"3. **Wait for Better Price**: Target price of {better_price:.1%} for larger edge")

    sections.append("")
    return sections


def _build_final_recommendation(should_bet: bool, rec_size: float, size_pct: float,
                               edge_pct: float, ev_pct: float, execution_risk: str,
                               portfolio_risk: str, reasoning: str) -> List[str]:
    """Build final recommendation section."""
    sections = ["## FINAL RECOMMENDATION"]

    if should_bet and portfolio_risk != "high" and execution_risk != "high":
        recommendation = f"**✅ BET YES** - ${rec_size:,.2f} ({size_pct:.1f}% of bankroll)"
        confidence_level = "HIGH" if edge_pct >= 10 else "MEDIUM" if edge_pct >= 5 else "LOW"
        sections.extend([
            recommendation,
            f"**Confidence**: {confidence_level}",
            f"**Reasoning**: {edge_pct:+.1f}% edge with {ev_pct:+.1f}% EV. Market has acceptable liquidity and portfolio risk is managed."
        ])
    elif should_bet:
        sections.extend([
            f"**⚠️ BET WITH CAUTION** - Reduce size to ${rec_size * 0.5:,.2f}",
            f"**Reasoning**: Positive edge but {execution_risk} execution risk or {portfolio_risk} portfolio risk"
        ])
    else:
        sections.extend([
            "**❌ DO NOT BET**",
            f"**Reasoning**: {reasoning}"
        ])

    sections.extend(["", "---", "*NOT FINANCIAL ADVICE. This is research and analysis only.*"])
    return sections


@tool
def generate_bet_rationale_tool(
    market_question: str,
    p_bayesian: float,
    market_price: float,
    edge_analysis: Dict[str, Any],
    market_analysis: Dict[str, Any],
    portfolio_analysis: Dict[str, Any],
    position_size: Dict[str, Any],
    top_evidence: List[Dict[str, Any]] = None
) -> str:
    """
    Generate detailed human-readable rationale for betting recommendation.

    Synthesizes all analysis into clear reasoning covering: why bet (or not),
    key evidence, risks, alternative scenarios, and final recommendation.

    Args:
        market_question: The market question
        p_bayesian: Bayesian probability estimate
        market_price: Current market price
        edge_analysis: Results from evaluate_betting_edge_tool
        market_analysis: Results from analyze_market_context_tool
        portfolio_analysis: Results from assess_portfolio_risk_tool
        position_size: Results from calculate_position_size_tool
        top_evidence: Top 3-5 pieces of supporting evidence

    Returns:
        Formatted rationale string with sections

    Raises:
        AssertionError: If required analysis data is missing or invalid
    """
    # Validate inputs
    assert 0 <= p_bayesian <= 1, f"Bayesian probability must be between 0-1: {p_bayesian}"
    assert 0 <= market_price <= 1, f"Market price must be between 0-1: {market_price}"
    assert edge_analysis, "Edge analysis results required"
    assert market_analysis, "Market analysis results required"
    assert portfolio_analysis, "Portfolio analysis results required"
    assert position_size, "Position size results required"

    # Extract key metrics
    metrics = _extract_rationale_metrics(edge_analysis, market_analysis, portfolio_analysis, position_size)

    # Build rationale sections
    sections = []

    # Market assessment
    sections.extend(_build_market_assessment_section(
        market_question, p_bayesian, market_price,
        metrics["edge_pct"], metrics["ev_pct"],
        edge_analysis.get("value_assessment", "unknown")
    ))

    # Key evidence
    sections.extend(_build_evidence_section(top_evidence))

    # Market liquidity
    sections.extend(_build_liquidity_section(
        metrics["market_quality"], metrics["liquidity"],
        metrics["execution_risk"],
        market_analysis.get("spread", 0),
        market_analysis.get("spread_quality", "unknown")
    ))

    # Portfolio impact
    sections.extend(_build_portfolio_section(
        metrics["portfolio_risk"],
        portfolio_analysis.get("correlation_risk", "unknown"),
        portfolio_analysis.get("concentration_risk", "unknown"),
        metrics["warnings"]
    ))

    # Position sizing
    sections.extend(_build_position_section(
        metrics["should_bet"], metrics["rec_size"], metrics["size_pct"],
        metrics["kelly_pct"], position_size.get("size_category", "unknown"),
        position_size.get("rationale", ""),
        edge_analysis.get("reasoning", "Insufficient edge")
    ))

    # Key risks
    sections.append("## KEY RISKS")
    sections.extend(_identify_key_risks(
        metrics["execution_risk"], metrics["portfolio_risk"],
        metrics["edge_pct"], market_price
    ))
    sections.append("")

    # Alternatives
    sections.extend(_build_alternatives_section(metrics["edge_pct"], metrics["rec_size"], market_price))

    # Final recommendation
    sections.extend(_build_final_recommendation(
        metrics["should_bet"], metrics["rec_size"], metrics["size_pct"],
        metrics["edge_pct"], metrics["ev_pct"], metrics["execution_risk"],
        metrics["portfolio_risk"], edge_analysis.get("reasoning", "Insufficient edge or high risk")
    ))

    rationale = "\n".join(sections)

    logger.info(f"Generated betting rationale: {len(rationale)} chars, should_bet={metrics['should_bet']}")

    return rationale
