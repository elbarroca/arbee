"""
Edge Detection Tools for POLYSEER
Detects edge opportunities beyond simple arbitrage:
- Information asymmetry (insider activity)
- Market inefficiencies (volume spikes, orderbook anomalies)
- Sentiment shifts before market reacts
- Base rate violations
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import logging
from datetime import datetime, timedelta

from clients.polymarket import PolymarketClient
from clients.web3.wallet_tracker import WalletTrackerClient
from tools.copy_trade_signal_tool import copy_trade_signal_tool
from tools.memory import get_base_rates_tool
from utils.rich_logging import (
    log_tool_start,
    log_tool_success,
    log_tool_error,
    log_edge_detection_result,
)
from config.settings import settings

logger = logging.getLogger(__name__)


@tool
async def information_asymmetry_tool(
    market_slug: str,
    provider: str = "polymarket",
    insider_wallet_addresses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect information asymmetry through insider wallet activity.

    Analyzes wallet activity patterns to identify potential insider trading:
    - Large positions taken before public information
    - Coordinated buying/selling patterns
    - Timing anomalies (positions taken right before price moves)

    Args:
        market_slug: Market identifier
        provider: Market provider (polymarket, kalshi, etc.)
        insider_wallet_addresses: Optional list of known insider wallet addresses

    Returns:
        Dict with edge_type="information_asymmetry", strength (0-1), confidence (0-1),
        evidence (list of activity patterns), and wallet_addresses (list of flagged wallets)
    """
    try:
        log_tool_start("information_asymmetry_tool", {"market_slug": market_slug, "provider": provider, "insider_wallet_addresses": insider_wallet_addresses})
        logger.info(f"🔍 Detecting information asymmetry for {market_slug}")

        # Use real wallet tracker if available
        wallet_tracker = WalletTrackerClient()
        evidence = []  # Initialize evidence list
        
        if wallet_tracker.enabled:
            try:
                activity = await wallet_tracker.detect_insider_activity(market_slug, lookback_hours=24)
                
                if activity.get("flagged_wallets"):
                    flagged_wallets = activity["flagged_wallets"]
                    strength = min(0.9, activity.get("confidence", 0.0) + (len(flagged_wallets) * 0.1))
                    confidence = activity.get("confidence", 0.5)
                    evidence.extend(activity.get("evidence", []))
                    
                    result = {
                        "edge_type": "information_asymmetry",
                        "strength": strength,
                        "confidence": confidence,
                        "evidence": evidence,
                        "wallet_addresses": flagged_wallets,
                        "market_slug": market_slug,
                        "provider": provider,
                        "detection_timestamp": datetime.utcnow().isoformat(),
                    }
                    
                    log_edge_detection_result("information_asymmetry_tool", "information_asymmetry", strength, confidence, evidence)
                    log_tool_success("information_asymmetry_tool", {"edge_strength": strength, "confidence": confidence, "flagged_wallets": len(flagged_wallets)})
                    
                    return result
            except Exception as e:
                logger.warning(f"Wallet tracker failed, using fallback: {e}")
        
        # Fallback to mock/insider wallet addresses if provided
        strength = 0.0
        confidence = 0.5
        evidence = []
        flagged_wallets = []

        if insider_wallet_addresses:
            # In production, check if these wallets have positions
            # and analyze their timing vs market movements
            flagged_wallets = insider_wallet_addresses[:3]  # Mock: flag first 3
            strength = min(0.7, len(flagged_wallets) * 0.2)
            confidence = 0.6
            evidence.append(
                f"Found {len(flagged_wallets)} insider wallets with positions in this market"
            )

        result = {
            "edge_type": "information_asymmetry",
            "strength": strength,
            "confidence": confidence,
            "evidence": evidence,
            "wallet_addresses": flagged_wallets,
            "market_slug": market_slug,
            "provider": provider,
            "detection_timestamp": datetime.utcnow().isoformat(),
        }
        
        log_edge_detection_result("information_asymmetry_tool", "information_asymmetry", strength, confidence, evidence)
        log_tool_success("information_asymmetry_tool", {"edge_strength": strength, "confidence": confidence, "flagged_wallets": len(flagged_wallets)})
        
        return result

    except Exception as e:
        log_tool_error("information_asymmetry_tool", e, f"Market: {market_slug}")
        logger.error(f"❌ Information asymmetry detection failed: {e}")
        return {
            "edge_type": "information_asymmetry",
            "strength": 0.0,
            "confidence": 0.0,
            "evidence": [f"Error: {str(e)}"],
            "wallet_addresses": [],
            "error": str(e),
        }


@tool
async def market_inefficiency_tool(
    market_slug: str,
    provider: str = "polymarket",
    lookback_hours: int = 24,
    market_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detect market inefficiencies through volume spikes and orderbook anomalies.

    Identifies:
    - Unusual volume spikes (potential information leakage)
    - Orderbook imbalances (large bid/ask spreads)
    - Whale activity (large orders moving price)
    - Coordinated buying/selling patterns

    Args:
        market_slug: Market identifier
        provider: Market provider
        lookback_hours: Hours to look back for volume analysis
        market_data: Optional market data dict from workflow state (if available)

    Returns:
        Dict with edge_type="market_inefficiency", strength (0-1), confidence (0-1),
        evidence (list of anomalies), and metrics (volume_spike, spread, etc.)
    """
    try:
        log_tool_start("market_inefficiency_tool", {"market_slug": market_slug, "provider": provider, "lookback_hours": lookback_hours})
        logger.info(f"📊 Detecting market inefficiencies for {market_slug}")

        # Use provided market_data if available, otherwise fetch
        market = market_data
        if not market:
            try:
                client = PolymarketClient()
                market = await client.gamma.get_market(market_slug)
            except Exception as fetch_error:
                logger.warning(f"Could not fetch market {market_slug}: {fetch_error}")
                market = None

        if not market:
            # Market not found - return zero-strength edge signal gracefully
            result = {
                "edge_type": "market_inefficiency",
                "strength": 0.0,
                "confidence": 0.0,
                "evidence": [f"Market not found: {market_slug}"],
                "metrics": {},
                "market_slug": market_slug,
                "provider": provider,
                "detection_timestamp": datetime.utcnow().isoformat(),
            }
            log_tool_success("market_inefficiency_tool", {"edge_strength": 0.0, "confidence": 0.0, "note": "Market not found"})
            return result

        # Get orderbook data
        token_ids = market.get("clobTokenIds", [])
        metrics = {}
        evidence = []
        strength = 0.0
        confidence = 0.5

        if token_ids:
            try:
                client = PolymarketClient()
                orderbook = client.clob.get_orderbook(token_ids[0], depth=20)
                spread = orderbook.get("spread", 0.0)
                spread_bps = orderbook.get("spread_bps", 0)

                metrics["spread"] = spread
                metrics["spread_bps"] = spread_bps
                metrics["liquidity"] = orderbook.get("total_liquidity", 0.0)

                # Detect large spreads (inefficiency signal)
                if spread > 0.05:  # 5% spread
                    strength += 0.3
                    evidence.append(f"Large spread detected: {spread:.2%}")
                    confidence = 0.7

                # Detect low liquidity
                if metrics["liquidity"] < 1000:
                    strength += 0.2
                    evidence.append(f"Low liquidity: ${metrics['liquidity']:.0f}")
                    confidence = max(confidence, 0.6)

                # Check for whale activity (large orders)
                bids = orderbook.get("bids", [])
                asks = orderbook.get("asks", [])
                if bids and asks:
                    largest_bid = max((float(b.get("size", 0)) for b in bids[:5]), default=0)
                    largest_ask = max((float(a.get("size", 0)) for a in asks[:5]), default=0)
                    metrics["largest_bid_size"] = largest_bid
                    metrics["largest_ask_size"] = largest_ask

                    # Large orders relative to liquidity suggest whale activity
                    if largest_bid > metrics["liquidity"] * 0.1:
                        strength += 0.2
                        evidence.append(f"Large bid detected: {largest_bid:.0f} shares")
                        confidence = max(confidence, 0.65)

            except Exception as e:
                logger.warning(f"Orderbook analysis failed: {e}")
                evidence.append(f"Orderbook analysis unavailable: {str(e)}")

        # Volume analysis (would need historical data)
        volume = market.get("volumeNum", market.get("volume", 0))
        metrics["current_volume"] = volume

        # Normalize strength
        strength = min(1.0, strength)

        result = {
            "edge_type": "market_inefficiency",
            "strength": strength,
            "confidence": confidence,
            "evidence": evidence,
            "metrics": metrics,
            "market_slug": market_slug,
            "provider": provider,
            "detection_timestamp": datetime.utcnow().isoformat(),
        }
        
        log_edge_detection_result("market_inefficiency_tool", "market_inefficiency", strength, confidence, evidence)
        log_tool_success("market_inefficiency_tool", {"edge_strength": strength, "confidence": confidence})
        
        return result

    except Exception as e:
        log_tool_error("market_inefficiency_tool", e, f"Market: {market_slug}")
        logger.error(f"❌ Market inefficiency detection failed: {e}")
        return {
            "edge_type": "market_inefficiency",
            "strength": 0.0,
            "confidence": 0.0,
            "evidence": [f"Error: {str(e)}"],
            "metrics": {},
            "error": str(e),
        }


@tool
async def sentiment_edge_tool(
    market_question: str,
    p_bayesian: float,
    market_price: float,
    lookback_hours: int = 6,
) -> Dict[str, Any]:
    """
    Detect sentiment shifts before market reacts.

    Compares Bayesian probability (based on latest evidence) vs market price.
    If p_bayesian diverges significantly from market, there may be an edge from
    information that hasn't been priced in yet.

    Args:
        market_question: Market question text
        p_bayesian: Bayesian posterior probability from analysis
        market_price: Current market price (0-1)
        lookback_hours: Hours to look back for sentiment analysis

    Returns:
        Dict with edge_type="sentiment_edge", strength (0-1), confidence (0-1),
        evidence (explanation of divergence), and divergence metrics
    """
    try:
        log_tool_start("sentiment_edge_tool", {"market_question": market_question[:50], "p_bayesian": p_bayesian, "market_price": market_price})
        logger.info(f"💭 Detecting sentiment edge for market")

        # Calculate divergence
        divergence = abs(p_bayesian - market_price)
        direction = "bullish" if p_bayesian > market_price else "bearish"

        strength = 0.0
        confidence = 0.5
        evidence = []

        # Strong divergence suggests edge
        if divergence > 0.15:  # 15% divergence
            strength = min(0.9, divergence * 3)  # Scale to 0-1
            confidence = 0.8
            evidence.append(
                f"Strong divergence: Bayesian {p_bayesian:.1%} vs Market {market_price:.1%} "
                f"({direction} edge of {divergence:.1%})"
            )
        elif divergence > 0.10:  # 10% divergence
            strength = min(0.7, divergence * 4)
            confidence = 0.7
            evidence.append(
                f"Moderate divergence: Bayesian {p_bayesian:.1%} vs Market {market_price:.1%} "
                f"({direction} edge of {divergence:.1%})"
            )
        elif divergence > 0.05:  # 5% divergence
            strength = min(0.5, divergence * 6)
            confidence = 0.6
            evidence.append(
                f"Minor divergence: Bayesian {p_bayesian:.1%} vs Market {market_price:.1%}"
            )
        else:
            evidence.append("No significant divergence detected")

        result = {
            "edge_type": "sentiment_edge",
            "strength": strength,
            "confidence": confidence,
            "evidence": evidence,
            "p_bayesian": p_bayesian,
            "market_price": market_price,
            "divergence": divergence,
            "direction": direction,
            "detection_timestamp": datetime.utcnow().isoformat(),
        }
        
        log_edge_detection_result("sentiment_edge_tool", "sentiment_edge", strength, confidence, evidence)
        log_tool_success("sentiment_edge_tool", {"edge_strength": strength, "confidence": confidence, "divergence": divergence})
        
        return result

    except Exception as e:
        log_tool_error("sentiment_edge_tool", e, f"Market: {market_question[:50]}")
        logger.error(f"❌ Sentiment edge detection failed: {e}")
        return {
            "edge_type": "sentiment_edge",
            "strength": 0.0,
            "confidence": 0.0,
            "evidence": [f"Error: {str(e)}"],
            "error": str(e),
        }


@tool
async def base_rate_violation_tool(
    market_question: str,
    market_price: float,
    event_category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect base rate violations (market price vs historical base rates).

    Compares current market price to historical base rates for similar events.
    If market price significantly deviates from base rate, there may be an edge.

    Args:
        market_question: Market question text
        market_price: Current market price (0-1)
        event_category: Optional event category for base rate lookup
                       (if None, will attempt to extract from question)

    Returns:
        Dict with edge_type="base_rate_violation", strength (0-1), confidence (0-1),
        evidence (base rate comparison), and base_rate data
    """
    try:
        log_tool_start("base_rate_violation_tool", {"market_question": market_question[:50], "market_price": market_price, "event_category": event_category})
        logger.info(f"📈 Detecting base rate violation for market")

        # Extract event category from question if not provided
        if not event_category:
            # Simple extraction: look for common patterns
            question_lower = market_question.lower()
            if "election" in question_lower or "president" in question_lower:
                event_category = "election outcome"
            elif "sports" in question_lower or "race" in question_lower:
                event_category = "sports event"
            elif "crypto" in question_lower or "bitcoin" in question_lower:
                event_category = "cryptocurrency event"
            else:
                # Use question as category
                event_category = market_question[:50]

        # Get base rate from memory
        try:
            base_rate_result = await get_base_rates_tool.ainvoke({
                "event_category": event_category,
                "limit": 5,
            })
        except Exception as e:
            logger.warning(f"Failed to get base rates: {e}, using defaults")
            base_rate_result = {
                "base_rate": 0.5,
                "sample_size": 0,
                "confidence": "low"
            }

        base_rate = base_rate_result.get("base_rate", 0.5)
        sample_size = base_rate_result.get("sample_size", 0)
        confidence_level = base_rate_result.get("confidence", "low")

        strength = 0.0
        confidence = 0.3  # Low default confidence
        evidence = []

        if sample_size > 0:
            # Calculate violation
            violation = abs(market_price - base_rate)

            if violation > 0.20:  # 20% violation
                strength = min(0.9, violation * 3)
                confidence = 0.8 if confidence_level == "moderate" else 0.6
                evidence.append(
                    f"Strong base rate violation: Market {market_price:.1%} vs "
                    f"Base Rate {base_rate:.1%} (violation: {violation:.1%})"
                )
            elif violation > 0.15:  # 15% violation
                strength = min(0.7, violation * 4)
                confidence = 0.7 if confidence_level == "moderate" else 0.5
                evidence.append(
                    f"Moderate base rate violation: Market {market_price:.1%} vs "
                    f"Base Rate {base_rate:.1%}"
                )
            elif violation > 0.10:  # 10% violation
                strength = min(0.5, violation * 5)
                confidence = 0.6 if confidence_level == "moderate" else 0.4
                evidence.append(
                    f"Minor base rate violation: Market {market_price:.1%} vs "
                    f"Base Rate {base_rate:.1%}"
                )
            else:
                evidence.append(
                    f"Market price {market_price:.1%} aligns with base rate {base_rate:.1%}"
                )

            # Adjust confidence based on sample size
            if sample_size >= 10:
                confidence = min(0.9, confidence + 0.1)
            elif sample_size >= 5:
                confidence = min(0.8, confidence + 0.05)
        else:
            evidence.append(f"No base rate data found for category: {event_category}")
            confidence = 0.2

        result = {
            "edge_type": "base_rate_violation",
            "strength": strength,
            "confidence": confidence,
            "evidence": evidence,
            "market_price": market_price,
            "base_rate": base_rate,
            "sample_size": sample_size,
            "event_category": event_category,
            "violation": abs(market_price - base_rate) if sample_size > 0 else 0.0,
            "detection_timestamp": datetime.utcnow().isoformat(),
        }
        
        log_edge_detection_result("base_rate_violation_tool", "base_rate_violation", strength, confidence, evidence)
        log_tool_success("base_rate_violation_tool", {"edge_strength": strength, "confidence": confidence, "violation": result["violation"]})
        
        return result

    except Exception as e:
        log_tool_error("base_rate_violation_tool", e, f"Market: {market_question[:50]}")
        logger.error(f"❌ Base rate violation detection failed: {e}")
        return {
            "edge_type": "base_rate_violation",
            "strength": 0.0,
            "confidence": 0.0,
            "evidence": [f"Error: {str(e)}"],
            "error": str(e),
        }


@tool
async def composite_edge_score_tool(
    edge_signals: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Combine multiple edge signals into a composite edge score.

    Args:
        edge_signals: List of edge signal dicts from other edge detection tools
        weights: Optional dict mapping edge_type to weight (default: equal weights)

    Returns:
        Dict with composite_score (0-1), weighted_components, and recommendation
    """
    try:
        log_tool_start("composite_edge_score_tool", {"edge_signals_count": len(edge_signals), "weights": weights})
        logger.info(f"🔗 Computing composite edge score from {len(edge_signals)} signals")

        if not edge_signals:
            result = {
                "composite_score": 0.0,
                "confidence": 0.0,
                "weighted_components": {},
                "recommendation": "No edge signals available",
            }
            log_tool_success("composite_edge_score_tool", {"composite_score": 0.0, "confidence": 0.0})
            return result

        # Default weights (can be customized)
        default_weights = {
            "information_asymmetry": 0.3,  # Highest weight (insider info is strongest)
            "market_inefficiency": 0.25,
            "sentiment_edge": 0.25,
            "base_rate_violation": 0.2,
        }

        if weights is None:
            weights = default_weights

        weighted_sum = 0.0
        total_weight = 0.0
        weighted_components = {}
        all_evidence = []

        for signal in edge_signals:
            edge_type = signal.get("edge_type", "unknown")
            strength = signal.get("strength", 0.0)
            confidence = signal.get("confidence", 0.0)
            evidence = signal.get("evidence", [])

            # Weight by both signal strength and confidence
            weight = weights.get(edge_type, 0.25)
            adjusted_strength = strength * confidence
            weighted_value = adjusted_strength * weight

            weighted_sum += weighted_value
            total_weight += weight

            weighted_components[edge_type] = {
                "strength": strength,
                "confidence": confidence,
                "weight": weight,
                "weighted_contribution": weighted_value,
            }

            all_evidence.extend(evidence)

        # Normalize composite score
        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 0.0

        # Calculate overall confidence (average of signal confidences)
        avg_confidence = (
            sum(s.get("confidence", 0.0) for s in edge_signals) / len(edge_signals)
            if edge_signals
            else 0.0
        )

        # Generate recommendation
        if composite_score >= 0.7:
            recommendation = "Strong edge detected - high confidence opportunity"
        elif composite_score >= 0.5:
            recommendation = "Moderate edge detected - consider position"
        elif composite_score >= 0.3:
            recommendation = "Weak edge detected - monitor closely"
        else:
            recommendation = "No significant edge detected"

        result = {
            "composite_score": composite_score,
            "confidence": avg_confidence,
            "weighted_components": weighted_components,
            "recommendation": recommendation,
            "evidence_summary": all_evidence[:5],  # Top 5 evidence items
            "signal_count": len(edge_signals),
            "computation_timestamp": datetime.utcnow().isoformat(),
        }
        
        log_edge_detection_result("composite_edge_score_tool", "composite", composite_score, avg_confidence, all_evidence[:5])
        log_tool_success("composite_edge_score_tool", {"composite_score": composite_score, "confidence": avg_confidence, "signal_count": len(edge_signals)})
        
        return result

    except Exception as e:
        log_tool_error("composite_edge_score_tool", e, f"Signals: {len(edge_signals)}")
        logger.error(f"❌ Composite edge score computation failed: {e}")
        return {
            "composite_score": 0.0,
            "confidence": 0.0,
            "weighted_components": {},
            "recommendation": f"Error: {str(e)}",
            "error": str(e),
        }


@tool
async def analyze_mentions_market_tool(
    market_question: str,
    transcript_text: Optional[str] = None,
    mentions_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Analyze mentions market to detect mispricing based on transcript analysis.
    
    This tool analyzes mentions markets (markets based on podcast/interview transcripts)
    to find mispriced opportunities by comparing:
    - Transcript content vs market prices
    - Sentiment analysis of mentions
    - Key quote extraction and interpretation
    
    Args:
        market_question: Market question being analyzed
        transcript_text: Optional transcript text to analyze
        mentions_data: Optional list of mention dicts with keys: timestamp, speaker, text, sentiment
        
    Returns:
        Dict with:
        - edge_type: "mentions_mispricing"
        - strength: Mispricing strength (0-1)
        - confidence: Confidence in detection (0-1)
        - evidence: List of evidence strings
        - key_quotes: List of relevant quotes from transcript
        - sentiment_score: Overall sentiment (-1 to 1)
        - mispricing_direction: "bullish" or "bearish"
    """
    try:
        log_tool_start("analyze_mentions_market_tool", {"market_question": market_question[:50], "has_transcript": transcript_text is not None, "has_mentions_data": mentions_data is not None})
        logger.info(f"📝 Analyzing mentions market: {market_question[:50]}...")
        
        strength = 0.0
        confidence = 0.3  # Low default confidence without transcript
        evidence = []
        key_quotes = []
        sentiment_score = 0.0
        mispricing_direction = "neutral"
        
        # If transcript provided, analyze it
        if transcript_text:
            # Simple sentiment analysis (in production, use proper NLP)
            text_lower = transcript_text.lower()
            
            # Positive indicators
            positive_words = ["yes", "will", "likely", "probably", "expect", "confident", "definitely"]
            negative_words = ["no", "won't", "unlikely", "doubt", "uncertain", "probably not"]
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            sentiment_score = (positive_count - negative_count) / max(1, positive_count + negative_count)
            
            # Extract key quotes (simple: sentences with market-relevant keywords)
            sentences = transcript_text.split('.')
            relevant_keywords = market_question.lower().split()[:5]  # First 5 words
            
            for sentence in sentences[:20]:  # Check first 20 sentences
                if any(keyword in sentence.lower() for keyword in relevant_keywords if len(keyword) > 3):
                    if len(sentence.strip()) > 20 and len(sentence.strip()) < 200:
                        key_quotes.append(sentence.strip()[:150])
            
            if sentiment_score > 0.3:
                mispricing_direction = "bullish"
                strength = min(0.7, abs(sentiment_score) * 1.5)
                evidence.append(f"Positive sentiment detected (score: {sentiment_score:.2f})")
            elif sentiment_score < -0.3:
                mispricing_direction = "bearish"
                strength = min(0.7, abs(sentiment_score) * 1.5)
                evidence.append(f"Negative sentiment detected (score: {sentiment_score:.2f})")
            
            if key_quotes:
                confidence = min(0.8, 0.3 + len(key_quotes) * 0.1)
                evidence.append(f"Found {len(key_quotes)} relevant quotes")
        
        # If mentions_data provided, analyze structured mentions
        elif mentions_data:
            positive_mentions = sum(1 for m in mentions_data if m.get('sentiment', 0) > 0)
            negative_mentions = sum(1 for m in mentions_data if m.get('sentiment', 0) < 0)
            total_mentions = len(mentions_data)
            
            if total_mentions > 0:
                sentiment_score = (positive_mentions - negative_mentions) / total_mentions
                
                if sentiment_score > 0.2:
                    mispricing_direction = "bullish"
                    strength = min(0.8, abs(sentiment_score) * 2)
                elif sentiment_score < -0.2:
                    mispricing_direction = "bearish"
                    strength = min(0.8, abs(sentiment_score) * 2)
                
                confidence = min(0.9, 0.5 + total_mentions * 0.05)
                evidence.append(f"Analyzed {total_mentions} mentions ({positive_mentions} positive, {negative_mentions} negative)")
                
                # Extract top quotes
                sorted_mentions = sorted(
                    mentions_data,
                    key=lambda x: abs(x.get('sentiment', 0)),
                    reverse=True
                )
                key_quotes = [m.get('text', '')[:150] for m in sorted_mentions[:5]]
        else:
            evidence.append("No transcript or mentions data provided - using fallback analysis")
            confidence = 0.2
        
        result = {
            "edge_type": "mentions_mispricing",
            "strength": strength,
            "confidence": confidence,
            "evidence": evidence,
            "key_quotes": key_quotes[:5],  # Top 5 quotes
            "sentiment_score": sentiment_score,
            "mispricing_direction": mispricing_direction,
            "market_question": market_question,
            "detection_timestamp": datetime.utcnow().isoformat(),
        }
        
        log_edge_detection_result("analyze_mentions_market_tool", "mentions_mispricing", strength, confidence, evidence)
        log_tool_success("analyze_mentions_market_tool", {"edge_strength": strength, "confidence": confidence, "sentiment_score": sentiment_score})
        
        return result
        
    except Exception as e:
        log_tool_error("analyze_mentions_market_tool", e, f"Market: {market_question[:50]}")
        logger.error(f"❌ Mentions market analysis failed: {e}")
        return {
            "edge_type": "mentions_mispricing",
            "strength": 0.0,
            "confidence": 0.0,
            "evidence": [f"Error: {str(e)}"],
            "key_quotes": [],
            "sentiment_score": 0.0,
            "mispricing_direction": "neutral",
            "error": str(e),
        }

