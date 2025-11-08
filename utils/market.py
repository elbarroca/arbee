"""
Market Matching and Pattern Detection Utilities
Finds similar markets across different prediction platforms for arbitrage analysis
and provides generic pattern detection for any market type.
"""
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

from api_clients.kalshi import KalshiClient
from api_clients.polymarket import PolymarketClient
from api_clients.valyu import ValyuResearchClient

logger = logging.getLogger(__name__)


# ============================================================================
# Market Matching (from market_matcher.py)
# ============================================================================

@dataclass
class MarketMatch:
    """Represents a matched pair of markets from different platforms"""
    polymarket_market: Dict[str, Any]
    kalshi_market: Dict[str, Any]
    similarity_score: float
    match_type: str  # "exact", "semantic", "category", "fuzzy"
    key_differences: Dict[str, Any]


@dataclass
class ArbitrageOpportunity:
    """Represents a potential arbitrage opportunity"""
    poly_price: float
    kalshi_price: float
    price_diff: float
    edge: float
    confidence: float
    recommended_action: str
    risk_factors: List[str]


class MarketMatcher:
    """Matches markets across prediction platforms for arbitrage analysis"""

    def __init__(self):
        self.poly_client = PolymarketClient()
        self.kalshi_client = KalshiClient()
        self.valyu_client = ValyuResearchClient()

    async def find_matching_markets(
        self,
        poly_limit: int = 50,
        kalshi_limit: int = 50,
        min_similarity: float = 0.6
    ) -> List[MarketMatch]:
        """
        Find markets that match between Polymarket and Kalshi

        Args:
            poly_limit: Max Polymarket markets to fetch
            kalshi_limit: Max Kalshi markets to fetch
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of matched market pairs
        """
        print("🔍 Fetching markets from both platforms...")

        # Fetch markets from both platforms
        poly_markets, kalshi_markets = await asyncio.gather(
            self.poly_client.get_event(limit=poly_limit, active=True),
            self.kalshi_client.get_event(limit=kalshi_limit)
        )

        assert poly_markets, "Polymarket data fetch failed"
        assert kalshi_markets, "Kalshi data fetch failed"
        print(f"📊 Retrieved {len(poly_markets)} Polymarket and {len(kalshi_markets)} Kalshi markets")

        # Normalize market data
        normalized_poly = self._normalize_polymarket_data(poly_markets)
        normalized_kalshi = self._normalize_kalshi_data(kalshi_markets)

        # Debug: Show sample normalized data
        assert normalized_poly, "Polymarket markets required"
        assert normalized_kalshi, "Kalshi markets required"
        print(f"📋 Sample Polymarket: {normalized_poly[0]['question']} (tags: {normalized_poly[0]['tags']})")
        print(f"📋 Sample Kalshi: {normalized_kalshi[0]['question']} (tags: {normalized_kalshi[0]['tags']})")

        # Find matches
        matches = []
        max_similarity = 0
        for poly_market in normalized_poly:
            for kalshi_market in normalized_kalshi:
                similarity = self._calculate_similarity(poly_market, kalshi_market)
                max_similarity = max(max_similarity, similarity)

                if similarity >= min_similarity:
                    match = MarketMatch(
                        polymarket_market=poly_market,
                        kalshi_market=kalshi_market,
                        similarity_score=similarity,
                        match_type=self._determine_match_type(poly_market, kalshi_market, similarity),
                        key_differences=self._find_differences(poly_market, kalshi_market)
                    )
                    matches.append(match)

        print(f"🎯 Max similarity found: {max_similarity:.3f}")
        print(f"🎯 Found {len(matches)} matching market pairs (threshold: {min_similarity})")

        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)

        return matches

    def _normalize_polymarket_data(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize Polymarket data for comparison"""
        assert markets, "Markets list required"
        normalized = []

        for market in markets:
            assert isinstance(market, dict), "Market must be dictionary"
            # Parse JSON fields
            if isinstance(market.get('outcomes'), str):
                market['outcomes'] = json.loads(market['outcomes'])

            if isinstance(market.get('outcomePrices'), str):
                market['outcomePrices'] = json.loads(market['outcomePrices'])
                market['outcomePrices'] = [float(p) for p in market['outcomePrices']]

            # Convert liquidity to float if string
            if isinstance(market.get('liquidity'), str):
                market['liquidity'] = float(market['liquidity'])

            # Extract key fields for comparison
            normalized_market = {
                'id': market.get('id', ''),
                'question': market.get('question', '').lower(),
                'category': market.get('category', '').lower(),
                'outcomes': [o.lower() for o in market.get('outcomes', [])],
                'end_date': market.get('endDate', ''),
                'liquidity': market.get('liquidityNum', 0),
                'volume': market.get('volumeNum', 0),
                'active': market.get('active', False),
                'description': market.get('description', '').lower(),
                'tags': self._extract_keywords(market.get('question', '') + ' ' + market.get('description', ''))
            }
            normalized.append(normalized_market)

        return normalized

    def _normalize_kalshi_data(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize Kalshi data for comparison"""
        assert markets, "Markets list required"
        normalized = []

        for market in markets:
            assert isinstance(market, dict), "Market must be dictionary"
            # Extract key fields for comparison
            normalized_market = {
                'id': market.get('ticker', ''),
                'question': market.get('title', '').lower(),
                'category': market.get('category', '').lower(),
                'outcomes': [
                    market.get('yes_sub_title', '').lower(),
                    market.get('no_sub_title', '').lower()
                ],
                'end_date': market.get('expiration_time', ''),
                'liquidity': market.get('liquidity', 0),
                'volume': market.get('volume', 0),
                'active': market.get('status') == 'active',
                'description': market.get('subtitle', '').lower(),
                'tags': self._extract_keywords(market.get('title', '') + ' ' + market.get('subtitle', ''))
            }
            normalized.append(normalized_market)

        return normalized

    def _calculate_similarity(self, market1: Dict[str, Any], market2: Dict[str, Any]) -> float:
        """Calculate similarity score between two markets (0-1)"""
        score = 0.0
        factors = 0

        # Question similarity (most important)
        question1 = market1.get('question', '')
        question2 = market2.get('question', '')
        assert question1 and question2, "Both markets must have questions"
        question_sim = self._text_similarity(question1, question2)
        score += question_sim * 0.4
        factors += 0.4

        # Category similarity
        cat1 = market1.get('category', '')
        cat2 = market2.get('category', '')
        cat_sim = 1.0 if cat1 == cat2 else 0.0
        score += cat_sim * 0.2
        factors += 0.2

        # Outcome similarity
        outcomes1 = set(market1.get('outcomes', []))
        outcomes2 = set(market2.get('outcomes', []))
        assert outcomes1 and outcomes2, "Both markets must have outcomes"
        outcome_overlap = len(outcomes1.intersection(outcomes2)) / max(len(outcomes1.union(outcomes2)), 1)
        score += outcome_overlap * 0.2
        factors += 0.2

        # Keyword similarity
        tags1 = set(market1.get('tags', []))
        tags2 = set(market2.get('tags', []))
        assert tags1 and tags2, "Both markets must have tags"
        tag_overlap = len(tags1.intersection(tags2)) / max(len(tags1.union(tags2)), 1)
        score += tag_overlap * 0.15
        factors += 0.15

        # Time proximity
        end_date1 = market1.get('end_date', '')
        end_date2 = market2.get('end_date', '')
        assert end_date1 and end_date2, "Both markets must have end dates"
        time_sim = self._time_similarity(end_date1, end_date2)
        score += time_sim * 0.02
        factors += 0.02

        assert factors > 0, "Similarity calculation must have at least one factor"
        return min(score / factors, 1.0)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods"""
        assert text1 and text2, "Both texts required for similarity calculation"

        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))
        assert words1 and words2, "Both texts must contain words"

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        assert union > 0, "Union of words must be non-empty"
        
        return overlap / union

    def _time_similarity(self, date1: str, date2: str) -> float:
        """Calculate similarity based on time proximity"""
        assert date1 and date2, "Both dates required for time similarity calculation"
        
        dt1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
        dt2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
        
        diff_days = abs((dt1 - dt2).days)
        return max(0, 1.0 - (diff_days / 30.0))

    def _determine_match_type(self, market1: Dict[str, Any], market2: Dict[str, Any], similarity: float) -> str:
        """Determine the type of match based on similarity and characteristics"""
        question1 = market1.get('question', '')
        question2 = market2.get('question', '')

        # Exact match
        if question1 == question2:
            return "exact"

        # Very high similarity with same category
        if similarity > 0.8 and market1.get('category') == market2.get('category'):
            return "semantic"

        # Same category with good keyword overlap
        if market1.get('category') == market2.get('category') and similarity > 0.6:
            return "category"

        # Lower similarity but some overlap
        return "fuzzy"

    def _find_differences(self, market1: Dict[str, Any], market2: Dict[str, Any]) -> Dict[str, Any]:
        """Find key differences between matched markets"""
        differences = {}

        # Liquidity difference
        liq1 = market1.get('liquidity', 0)
        liq2 = market2.get('liquidity', 0)
        assert liq1 > 0 and liq2 > 0, "Both markets must have liquidity data"
        differences['liquidity_ratio'] = max(liq1, liq2) / min(liq1, liq2)

        # Volume difference
        vol1 = market1.get('volume', 0)
        vol2 = market2.get('volume', 0)
        assert vol1 > 0 and vol2 > 0, "Both markets must have volume data"
        differences['volume_ratio'] = max(vol1, vol2) / min(vol1, vol2)

        # Time difference
        end_date1 = market1.get('end_date', '')
        end_date2 = market2.get('end_date', '')
        assert end_date1 and end_date2, "Both markets must have end dates"
        differences['time_similarity'] = self._time_similarity(end_date1, end_date2)

        return differences

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        assert text, "Text required for keyword extraction"

        # Remove common stop words and extract meaningful terms
        stop_words = {
            'will', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'what', 'when', 'where', 'how', 'why', 'who', 'which', 'yes', 'no', 'not', 'before', 'after',
            'during', 'between', 'among', 'through', 'over', 'under', 'above', 'below', 'around', 'behind'
        }

        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Also extract important named entities and numbers
        important_terms = []
        for word in words:
            # Keep numbers (years, dates, etc.)
            if word.isdigit() and len(word) == 4:  # Likely a year
                important_terms.append(word)
            # Keep capitalized words (proper nouns)
            elif word.istitle() and len(word) > 3:
                important_terms.append(word.lower())

        keywords.extend(important_terms)
        return list(set(keywords))[:15]  # Remove duplicates and limit to top 15

    async def analyze_arbitrage_opportunities(self, matches: List[MarketMatch]) -> List[ArbitrageOpportunity]:
        """Analyze arbitrage opportunities in matched markets"""
        opportunities = []

        for match in matches:
            # Get prices from both platforms
            poly_price = self._get_polymarket_price(match.polymarket_market)
            kalshi_price = await self._get_kalshi_price(match.kalshi_market)

            assert poly_price is not None, f"Polymarket price required for market {match.polymarket_market.get('id', 'unknown')}"
            assert kalshi_price is not None, f"Kalshi price required for market {match.kalshi_market.get('id', 'unknown')}"

            # Calculate arbitrage metrics
            price_diff = poly_price - kalshi_price
            edge = abs(price_diff)

            if edge > 0.02:  # 2% minimum edge
                opportunity = ArbitrageOpportunity(
                    poly_price=poly_price,
                    kalshi_price=kalshi_price,
                    price_diff=price_diff,
                    edge=edge,
                    confidence=match.similarity_score,
                    recommended_action="BUY" if price_diff > 0 else "SELL",
                    risk_factors=self._assess_risks(match)
                )
                opportunities.append(opportunity)

        # Sort by edge (highest first)
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        return opportunities

    def _get_polymarket_price(self, market: Dict[str, Any]) -> float:
        """Extract price from Polymarket data"""
        prices = market.get('outcomePrices', [])
        assert prices and len(prices) >= 2, f"Insufficient price data for market {market.get('id', 'unknown')}"
        return max(prices)

    async def _get_kalshi_price(self, market: Dict[str, Any]) -> float:
        """Extract price from Kalshi data"""
        assert 'id' in market, "Market ID required for Kalshi price"
        price = await self.kalshi_client.get_market_price(market['id'])
        assert price is not None, f"Price not found for Kalshi market {market['id']}"
        return price

    def _assess_risks(self, match: MarketMatch) -> List[str]:
        """Assess risks for a potential arbitrage trade"""
        risks = []

        # Liquidity risk
        poly_liq = match.polymarket_market.get('liquidity', 0)
        kalshi_liq = match.kalshi_market.get('liquidity', 0)

        if poly_liq < 1000 or kalshi_liq < 1000:
            risks.append("Low liquidity")

        # Time risk
        time_sim = match.key_differences.get('time_similarity', 1.0)
        if time_sim < 0.8:
            risks.append("Different expiration dates")

        # Platform risk
        if match.similarity_score < 0.7:
            risks.append("Low similarity between markets")

        return risks


# ============================================================================
# Market Pattern Detection (from market_patterns.py)
# ============================================================================

def extract_event_category(market_question: str) -> str:
    """
    Extract event category from market question (generic, not domain-specific).
    
    Args:
        market_question: Market question text
        
    Returns:
        Event category string (e.g., "election outcome", "sports event", etc.)
    """
    question_lower = market_question.lower()
    
    # Generic pattern matching (not domain-specific)
    patterns = {
        "election": ["election", "president", "vote", "ballot", "candidate", "campaign"],
        "sports": ["sports", "race", "game", "match", "championship", "tournament", "athlete"],
        "cryptocurrency": ["crypto", "bitcoin", "ethereum", "blockchain", "token", "coin"],
        "technology": ["tech", "ai", "software", "hardware", "product launch", "release"],
        "finance": ["stock", "market", "price", "earnings", "revenue", "profit"],
        "entertainment": ["movie", "film", "show", "celebrity", "award", "oscar"],
        "politics": ["policy", "law", "bill", "congress", "senate", "legislation"],
        "health": ["health", "medical", "disease", "treatment", "vaccine", "drug"],
    }
    
    for category, keywords in patterns.items():
        if any(keyword in question_lower for keyword in keywords):
            return f"{category} event"
    
    # Fallback: use first few words of question
    words = market_question.split()[:5]
    return " ".join(words).lower()[:50]


def find_similar_market_patterns(
    market_question: str,
    historical_markets: List[Dict[str, Any]],
    similarity_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Find similar markets based on question patterns (generic matching).
    
    Args:
        market_question: Current market question
        historical_markets: List of historical market dicts with 'question' field
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of similar markets with similarity scores
    """
    question_words = set(re.findall(r'\w+', market_question.lower()))
    similar = []
    
    for market in historical_markets:
        hist_question = market.get("question", "")
        hist_words = set(re.findall(r'\w+', hist_question.lower()))
        
        if not question_words or not hist_words:
            continue
        
        # Simple Jaccard similarity
        intersection = len(question_words & hist_words)
        union = len(question_words | hist_words)
        similarity = intersection / union if union > 0 else 0.0
        
        if similarity >= similarity_threshold:
            similar.append({
                **market,
                "similarity_score": similarity,
            })
    
    # Sort by similarity descending
    similar.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    return similar


def extract_key_variables(market_question: str) -> List[str]:
    """
    Extract key variables/factors from market question (generic extraction).
    
    Args:
        market_question: Market question text
        
    Returns:
        List of key variable strings
    """
    # Simple extraction: look for capitalized words, numbers, dates
    variables = []
    
    # Extract capitalized phrases (likely proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', market_question)
    variables.extend(capitalized)
    
    # Extract numbers (dates, amounts, etc.)
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', market_question)
    variables.extend(numbers)
    
    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', market_question)
    variables.extend(quoted)
    
    # Remove duplicates and return
    return list(set(variables))[:10]  # Limit to top 10


def categorize_market_type(market_question: str) -> Dict[str, Any]:
    """
    Categorize market type generically (not domain-specific).
    
    Args:
        market_question: Market question text
        
    Returns:
        Dict with category, subcategory, and confidence
    """
    category = extract_event_category(market_question)
    
    # Determine if binary or multi-outcome
    question_lower = market_question.lower()
    is_binary = any(word in question_lower for word in ["will", "does", "is", "has", "can"])
    
    # Determine timeframe
    timeframe = "unknown"
    if any(word in question_lower for word in ["2025", "2026", "this year", "next year"]):
        timeframe = "near_term"
    elif any(word in question_lower for word in ["by", "before", "after", "until"]):
        timeframe = "time_bound"
    else:
        timeframe = "open_ended"
    
    return {
        "category": category,
        "is_binary": is_binary,
        "timeframe": timeframe,
        "confidence": 0.7,  # Generic confidence
    }

