"""
PolyRouter Tools for Event Fetching and Market Search

Tools for fetching betting events and searching markets across multiple providers
using PolyRouter as the data aggregation source.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from utils.provider_manager import ProviderManager

logger = logging.getLogger(__name__)


@tool
async def fetch_events_from_polyrouter_tool(
    limit: int = 50,
    min_liquidity: float = 1000.0,
    include_orderbooks: bool = False,
    include_sports: bool = False,
) -> Dict[str, Any]:
    """
    Fetch betting events from PolyRouter (aggregates all providers).

    This tool fetches market data from PolyRouter which aggregates:
    - Polymarket
    - Kalshi
    - Manifold
    - Limitless
    - ProphetX
    - Novig
    - SX.bet

    Args:
        limit: Maximum number of markets to fetch (default: 50)
        min_liquidity: Minimum liquidity filter in USD (default: 1000)
        include_orderbooks: Include orderbook data (default: False, slower)
        include_sports: Include sports betting markets (default: False)

    Returns:
        Dict with:
        - markets: List of market dictionaries
        - providers: List of provider names
        - total_markets: Total count
        - summary_stats: Aggregate statistics

    Example:
        >>> result = await fetch_events_from_polyrouter_tool(limit=20, min_liquidity=10000)
        >>> print(f"Fetched {result['total_markets']} markets from {len(result['providers'])} providers")
    """
    try:
        manager = ProviderManager()

        logger.info(f"Fetching events from PolyRouter (limit={limit}, min_liquidity=${min_liquidity:,.0f})")

        # Get complete market data using ProviderManager
        data = await manager.get_complete_market_data(
            query=None,  # No query, get all markets
            limit=limit,
            min_liquidity=min_liquidity,
            include_orderbooks=include_orderbooks,
            include_sports=include_sports,
        )

        # Extract markets and providers
        markets = data.get("markets", [])
        summary_stats = data.get("summary_stats", {})
        providers = summary_stats.get("providers", [])

        result = {
            "markets": markets,
            "providers": providers,
            "total_markets": len(markets),
            "summary_stats": summary_stats,
        }

        logger.info(f"Successfully fetched {len(markets)} markets from {len(providers)} providers")

        return result

    except Exception as e:
        logger.error(f"Error fetching events from PolyRouter: {e}")
        return {
            "markets": [],
            "providers": [],
            "total_markets": 0,
            "summary_stats": {},
            "error": str(e),
        }


@tool
async def search_markets_across_providers_tool(
    market_question: str,
    limit: int = 20,
    min_liquidity: float = 1000.0,
) -> Dict[str, Any]:
    """
    Search for markets matching a specific question across all providers.

    Uses PolyRouter to search all connected providers for markets that match
    the given question. Useful for finding specific markets by name or topic.

    Args:
        market_question: Search query (e.g., "US Recession 2025", "Trump")
        limit: Maximum number of results (default: 20)
        min_liquidity: Minimum liquidity filter in USD (default: 1000)

    Returns:
        Dict with:
        - markets: List of matching market dictionaries
        - providers: List of providers with results
        - total_found: Total matches
        - best_match: Best matching market

    Example:
        >>> result = await search_markets_across_providers_tool(
        ...     market_question="US Recession 2025",
        ...     min_liquidity=10000
        ... )
        >>> for market in result['markets']:
        ...     print(f"{market['title']} on {market['platform']}")
    """
    try:
        manager = ProviderManager()

        logger.info(f"Searching for '{market_question}' across all providers")

        # Get complete market data with query
        data = await manager.get_complete_market_data(
            query=market_question,
            limit=limit,
            min_liquidity=min_liquidity,
            include_orderbooks=False,  # Faster without orderbooks
            include_sports=False,
        )

        # Extract markets and providers
        markets = data.get("markets", [])
        summary_stats = data.get("summary_stats", {})
        providers = summary_stats.get("providers", [])

        # Find best match (first result)
        best_match = markets[0] if markets else None

        result = {
            "markets": markets,
            "providers": providers,
            "total_found": len(markets),
            "best_match": best_match,
            "query": market_question,
        }

        logger.info(f"Found {len(markets)} markets matching '{market_question}' from {len(providers)} providers")

        return result

    except Exception as e:
        logger.error(f"Error searching markets for '{market_question}': {e}")
        return {
            "markets": [],
            "providers": [],
            "total_found": 0,
            "best_match": None,
            "query": market_question,
            "error": str(e),
        }


@tool
async def get_market_details_tool(
    market_slug: str,
    provider: str = "polymarket",
    include_orderbook: bool = False,
) -> Dict[str, Any]:
    """
    Get detailed information about a specific market.

    Fetches comprehensive market data including prices, volume, liquidity,
    and optionally orderbook data for execution analysis.

    Args:
        market_slug: Market slug/ID
        provider: Provider name (default: "polymarket")
        include_orderbook: Include orderbook data (default: False)

    Returns:
        Dict with market details including:
        - title: Market title
        - current_prices: YES/NO prices
        - volume: 24h trading volume
        - liquidity: Available liquidity
        - spread: Bid-ask spread
        - orderbook: Orderbook data (if requested)

    Example:
        >>> market = await get_market_details_tool(
        ...     market_slug="us-recession-2025",
        ...     include_orderbook=True
        ... )
        >>> print(f"Market: {market['title']}")
        >>> print(f"Spread: {market['spread']:.2%}")
    """
    try:
        from clients.polyrouter import PolyRouterClient
        import os

        api_key = os.getenv("POLYROUTER_API_KEY")
        client = PolyRouterClient(api_key=api_key)

        logger.info(f"Fetching details for market '{market_slug}' on {provider}")

        # Search for the market
        search_result = await client.search_markets(query=market_slug, limit=1)

        if not search_result or "markets" not in search_result:
            return {
                "error": f"Market '{market_slug}' not found",
                "market_slug": market_slug,
                "provider": provider,
            }

        markets = search_result.get("markets", [])
        if not markets:
            return {
                "error": f"No markets found matching '{market_slug}'",
                "market_slug": market_slug,
                "provider": provider,
            }

        # Get first matching market
        market = markets[0]

        # Add orderbook if requested
        if include_orderbook:
            try:
                orderbook = await client.get_orderbook(
                    market_id=market.get("market_id") or market.get("slug", market_slug),
                    provider=provider,
                )
                market["orderbook"] = orderbook
            except Exception as e:
                logger.warning(f"Failed to fetch orderbook: {e}")
                market["orderbook"] = None

        logger.info(f"Successfully fetched market details for '{market_slug}'")

        return market

    except Exception as e:
        logger.error(f"Error fetching market details for '{market_slug}': {e}")
        return {
            "error": str(e),
            "market_slug": market_slug,
            "provider": provider,
        }


@tool
async def find_arbitrage_opportunities_tool(
    markets: List[Dict[str, Any]],
    min_edge: float = 0.02,
    min_post_fee_margin: float = 0.01,
) -> Dict[str, Any]:
    """
    Find arbitrage opportunities across providers.

    Analyzes markets for cross-platform arbitrage (buy YES on A, NO on B)
    and single-platform mispricing opportunities.

    Args:
        markets: List of market dictionaries from PolyRouter
        min_edge: Minimum edge threshold (default: 2%)
        min_post_fee_margin: Minimum post-fee margin (default: 1%)

    Returns:
        Dict with:
        - arbitrage_opportunities: List of opportunities
        - cross_platform: Cross-platform arb opportunities
        - single_platform: Mispricing opportunities
        - total_opportunities: Count

    Example:
        >>> markets = await fetch_events_from_polyrouter_tool(limit=50)
        >>> arb = await find_arbitrage_opportunities_tool(
        ...     markets=markets['markets'],
        ...     min_edge=0.02
        ... )
        >>> print(f"Found {arb['total_opportunities']} arbitrage opportunities")
    """
    try:
        from clients.polyrouter_arbitrage import find_cross_platform_arbitrage

        logger.info(f"Finding arbitrage opportunities (min_edge={min_edge:.1%})")

        # Find cross-platform arbitrage
        opportunities = find_cross_platform_arbitrage(
            markets=markets,
            threshold=min_edge,
        )

        # Filter by post-fee margin
        filtered_opportunities = [
            opp for opp in opportunities
            if opp.get("post_fee_margin", 0) >= min_post_fee_margin
        ]

        # Categorize opportunities
        cross_platform = [
            opp for opp in filtered_opportunities
            if opp.get("type") == "cross_platform"
        ]

        single_platform = [
            opp for opp in filtered_opportunities
            if opp.get("type") == "single_platform"
        ]

        result = {
            "arbitrage_opportunities": filtered_opportunities,
            "cross_platform": cross_platform,
            "single_platform": single_platform,
            "total_opportunities": len(filtered_opportunities),
            "filters_applied": {
                "min_edge": min_edge,
                "min_post_fee_margin": min_post_fee_margin,
            },
        }

        logger.info(
            f"Found {len(filtered_opportunities)} arbitrage opportunities "
            f"({len(cross_platform)} cross-platform, {len(single_platform)} single-platform)"
        )

        return result

    except Exception as e:
        logger.error(f"Error finding arbitrage opportunities: {e}")
        return {
            "arbitrage_opportunities": [],
            "cross_platform": [],
            "single_platform": [],
            "total_opportunities": 0,
            "error": str(e),
        }


@tool
async def get_threshold_markets_tool(
    market_slug: str,
    provider: str = "polymarket",
) -> Dict[str, Any]:
    """
    Get related threshold markets (e.g., >50%, >60%, >70%).

    Threshold markets are variations of the same event with different outcome
    thresholds. Useful for finding better odds or constructing spreads.

    Currently only works for Polymarket.

    Args:
        market_slug: Market slug/ID
        provider: Provider name (default: "polymarket", only Polymarket supported)

    Returns:
        Dict with:
        - threshold_markets: List of related threshold markets
        - total_found: Count
        - base_market: Original market

    Example:
        >>> thresholds = await get_threshold_markets_tool(
        ...     market_slug="fed-rate-cuts-2025"
        ... )
        >>> for market in thresholds['threshold_markets']:
        ...     print(f"{market['title']}: {market['yes_price']:.1%}")
    """
    try:
        if provider.lower() != "polymarket":
            return {
                "error": f"Threshold markets only supported for Polymarket, not {provider}",
                "threshold_markets": [],
                "total_found": 0,
            }

        from clients.polymarket import PolymarketClient

        client = PolymarketClient()

        logger.info(f"Fetching threshold markets for '{market_slug}'")

        # Get threshold markets using Polymarket client
        threshold_markets = await client.get_threshold_markets(market_slug)

        result = {
            "threshold_markets": threshold_markets,
            "total_found": len(threshold_markets),
            "base_market_slug": market_slug,
            "provider": provider,
        }

        logger.info(f"Found {len(threshold_markets)} threshold markets for '{market_slug}'")

        return result

    except Exception as e:
        logger.error(f"Error fetching threshold markets for '{market_slug}': {e}")
        return {
            "error": str(e),
            "threshold_markets": [],
            "total_found": 0,
            "base_market_slug": market_slug,
            "provider": provider,
        }
