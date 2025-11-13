"""
Provider Manager Utility

Centralized manager for fetching market data from multiple providers
using PolyRouter as the primary data source.

Enhanced with:
- Complete market data retrieval across all 7 providers
- Advanced arbitrage detection (cross-platform, single-platform, multi-leg)
- Market quality scoring and analysis
- Sports betting support
"""
from typing import List, Dict, Any, Optional
import logging
from clients.polyrouter import PolyRouterClient
from clients.polymarket import PolymarketClient

# Optional Kalshi import
try:
    from clients.kalshi import KalshiClient
except ImportError:
    KalshiClient = None

from clients.polyrouter_sports import SportsBettingClient
from clients.polyrouter_arbitrage import (
    find_cross_platform_arbitrage,
    find_single_platform_mispricing,
    rank_opportunities,
)
from clients.polyrouter_analysis import (
    score_markets_batch,
    compare_providers,
    generate_recommendations,
)


logger = logging.getLogger(__name__)


class ProviderManager:
    """
    Unified interface for fetching market data from multiple providers.
    Uses PolyRouter for cross-platform aggregation.
    """

    def __init__(self, providers: Optional[List[str]] = None, api_key: Optional[str] = None):
        """
        Initialize provider manager.

        Args:
            providers: List of provider names to use (default: ["polymarket", "kalshi"])
            api_key: PolyRouter API key (optional, will use settings if not provided)
        """
        self.providers = providers or ["polymarket", "kalshi"]

        # Get API key from settings if not provided
        if api_key is None:
            from config.settings import settings
            import os
            api_key = os.getenv("POLYROUTER_API_KEY") or settings.POLYROUTER_API_KEY

        self.polyrouter = PolyRouterClient(api_key=api_key)

        # Initialize individual clients for direct access
        self.clients: Dict[str, Any] = {}
        if "polymarket" in self.providers:
            self.clients["polymarket"] = PolymarketClient()
        if "kalshi" in self.providers and KalshiClient is not None:
            self.clients["kalshi"] = KalshiClient()

        logger.info(f"ProviderManager initialized with providers: {self.providers}")

    async def get_all_events(
        self,
        limit: int = 50,
        offset: int = 0,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch betting events from all providers via PolyRouter.

        Args:
            limit: Maximum number of events to fetch
            offset: Pagination offset
            active_only: Only return active events

        Returns:
            List of events with market data
        """
        logger.info(f"Fetching {limit} events from all providers (offset={offset})")

        try:
            events = await self.polyrouter.get_events(limit=limit, offset=offset)

            if active_only:
                # Filter for active events only
                events = [e for e in events if e.get('active', True)]

            logger.info(f"Fetched {len(events)} events")
            return events

        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return []

    async def get_market_across_providers(
        self,
        question: str,
        limit: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for same market across all providers and group by provider.

        Args:
            question: Market question to search for
            limit: Maximum markets to fetch

        Returns:
            Dict mapping provider name to list of matching markets
        """
        logger.info(f"Searching for '{question}' across providers")

        try:
            markets = await self.polyrouter.search_markets(query=question, limit=limit)

            # Group by provider
            by_provider: Dict[str, List[Dict]] = {}
            for market in markets:
                provider = market.get('provider', 'unknown')
                if provider not in by_provider:
                    by_provider[provider] = []
                by_provider[provider].append(market)

            logger.info(f"Found markets from {len(by_provider)} providers")
            for provider, provider_markets in by_provider.items():
                logger.info(f"  {provider}: {len(provider_markets)} markets")

            return by_provider

        except Exception as e:
            logger.error(f"Error searching markets: {e}")
            return {}

    async def get_best_prices(
        self,
        markets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find best bid/ask prices across all provided markets.

        Args:
            markets: List of markets to analyze (from different providers)

        Returns:
            Dict with best YES and NO prices and their providers
        """
        if not markets:
            return {
                'best_yes_price': None,
                'best_yes_provider': None,
                'best_no_price': None,
                'best_no_provider': None
            }

        best_yes_price = float('inf')
        best_yes_provider = None
        best_no_price = float('inf')
        best_no_provider = None

        for market in markets:
            provider = market.get('provider', 'unknown')

            # Get YES price
            yes_price = market.get('yes_price') or market.get('price')
            if yes_price is not None and yes_price < best_yes_price:
                best_yes_price = yes_price
                best_yes_provider = provider

            # Get NO price (or calculate from YES if binary)
            no_price = market.get('no_price')
            if no_price is None and yes_price is not None:
                no_price = 1.0 - yes_price

            if no_price is not None and no_price < best_no_price:
                best_no_price = no_price
                best_no_provider = provider

        return {
            'best_yes_price': best_yes_price if best_yes_price != float('inf') else None,
            'best_yes_provider': best_yes_provider,
            'best_no_price': best_no_price if best_no_price != float('inf') else None,
            'best_no_provider': best_no_provider,
        }

    async def detect_arbitrage_opportunities(
        self,
        query: Optional[str] = None,
        markets: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.01,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Detect cross-platform arbitrage opportunities.

        Args:
            query: Market query to search for (if markets not provided)
            markets: Pre-fetched markets to analyze (optional)
            threshold: Minimum margin to consider an opportunity
            limit: Maximum markets to fetch if query is used

        Returns:
            List of arbitrage opportunities sorted by margin (descending)
        """
        # Fetch markets if not provided
        if markets is None:
            if query is None:
                logger.warning("Either query or markets must be provided")
                return []

            markets = await self.polyrouter.search_markets(query=query, limit=limit)

        logger.info(f"Analyzing {len(markets)} markets for arbitrage (threshold={threshold:.2%})")

        # Use PolyRouter's built-in arbitrage detection
        opportunities = self.polyrouter.find_arbitrage_opportunities(
            markets=markets,
            threshold=threshold
        )

        logger.info(f"Found {len(opportunities)} arbitrage opportunities")

        return opportunities

    async def get_threshold_markets(
        self,
        base_market_slug: str,
        base_question: str
    ) -> List[Dict[str, Any]]:
        """
        Find related threshold markets for a base market.
        Currently only supports Polymarket.

        Args:
            base_market_slug: Slug of the base market
            base_question: Question of the base market

        Returns:
            List of threshold markets with their data
        """
        logger.info(f"Finding threshold markets for: {base_market_slug}")

        if "polymarket" not in self.clients:
            logger.warning("Polymarket client not available for threshold market discovery")
            return []

        try:
            polymarket_client = self.clients["polymarket"]
            related = await polymarket_client.get_related_markets(
                base_market_slug,
                base_question
            )

            logger.info(f"Found {len(related)} threshold markets")
            return related

        except Exception as e:
            logger.error(f"Error finding threshold markets: {e}")
            return []

    async def get_enriched_market_data(
        self,
        market_slug: str,
        provider: str = "polymarket"
    ) -> Optional[Dict[str, Any]]:
        """
        Get enriched market data including orderbook, liquidity, spreads.

        Args:
            market_slug: Market slug/ID
            provider: Provider name (default: polymarket)

        Returns:
            Enriched market data or None if not found
        """
        logger.info(f"Fetching enriched data for {market_slug} from {provider}")

        if provider not in self.clients:
            logger.warning(f"Provider {provider} not available")
            return None

        try:
            client = self.clients[provider]

            if provider == "polymarket":
                # Get market data
                market = await client.gamma.get_market(market_slug)

                if not market:
                    logger.warning(f"Market {market_slug} not found")
                    return None

                # Enrich with orderbook
                token_ids = market.get('clobTokenIds', [])
                if token_ids and len(token_ids) >= 2:
                    yes_token_id = token_ids[1]
                    orderbook = client.clob.get_orderbook(yes_token_id, depth=10)

                    if orderbook:
                        market['orderbook'] = orderbook
                        market['mid_price'] = orderbook.get('mid_price')
                        market['spread'] = orderbook.get('spread')
                        market['spread_bps'] = int(orderbook.get('spread', 0) * 10000)

                return market

            elif provider == "kalshi":
                # Kalshi-specific enrichment
                market = await client.get_market(market_slug)
                return market

            else:
                logger.warning(f"Enrichment not implemented for {provider}")
                return None

        except Exception as e:
            logger.error(f"Error fetching enriched data: {e}")
            return None

    async def get_market_summary(
        self,
        question: str,
        include_threshold_markets: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive summary of a market across all providers.

        Args:
            question: Market question to search for
            include_threshold_markets: Include related threshold markets (Polymarket only)

        Returns:
            Dict with comprehensive market summary
        """
        logger.info(f"Getting market summary for: {question}")

        # Search across providers
        markets_by_provider = await self.get_market_across_providers(question, limit=20)

        if not markets_by_provider:
            return {
                'query': question,
                'found': False,
                'providers': [],
                'markets': [],
                'best_prices': None,
                'arbitrage_opportunities': [],
                'threshold_markets': []
            }

        # Flatten markets
        all_markets = []
        for provider_markets in markets_by_provider.values():
            all_markets.extend(provider_markets)

        # Get best prices
        best_prices = await self.get_best_prices(all_markets)

        # Detect arbitrage
        arbitrage_opportunities = await self.detect_arbitrage_opportunities(
            markets=all_markets,
            threshold=0.01
        )

        # Get threshold markets if requested
        threshold_markets = []
        if include_threshold_markets and all_markets:
            # Use first Polymarket market as base
            polymarket_markets = markets_by_provider.get('polymarket', [])
            if polymarket_markets:
                base_market = polymarket_markets[0]
                base_slug = base_market.get('slug', '')
                if base_slug:
                    threshold_markets = await self.get_threshold_markets(base_slug, question)

        summary = {
            'query': question,
            'found': True,
            'providers': list(markets_by_provider.keys()),
            'market_count': len(all_markets),
            'markets_by_provider': markets_by_provider,
            'best_prices': best_prices,
            'arbitrage_opportunities': arbitrage_opportunities,
            'threshold_markets': threshold_markets,
        }

        logger.info(
            f"Summary: {len(all_markets)} markets from {len(markets_by_provider)} providers, "
            f"{len(arbitrage_opportunities)} arbitrage opportunities, "
            f"{len(threshold_markets)} threshold markets"
        )

        return summary

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all configured providers.

        Returns:
            Dict mapping provider name to health status (bool)
        """
        logger.info("Running health check on all providers")

        health = {}

        # Check PolyRouter
        try:
            events = await self.polyrouter.get_events(limit=1)
            health['polyrouter'] = events is not None and len(events) > 0
        except Exception as e:
            logger.error(f"PolyRouter health check failed: {e}")
            health['polyrouter'] = False

        # Check individual providers
        for provider_name, client in self.clients.items():
            try:
                if provider_name == "polymarket":
                    markets = await client.gamma.get_markets(limit=1)
                    health[provider_name] = markets is not None and len(markets) > 0
                elif provider_name == "kalshi":
                    # Kalshi health check
                    # Implement based on Kalshi client API
                    health[provider_name] = True  # Placeholder
                else:
                    health[provider_name] = False
            except Exception as e:
                logger.error(f"{provider_name} health check failed: {e}")
                health[provider_name] = False

        logger.info(f"Health check results: {health}")
        return health

    # ==================== ENHANCED METHODS (Production-Ready) ==================== #

    async def get_complete_market_data(
        self,
        query: Optional[str] = None,
        limit: int = 100,
        min_liquidity: float = 1000.0,
        include_orderbooks: bool = False,
        include_sports: bool = False,
    ) -> Dict[str, Any]:
        """
        Get complete market data from all providers with comprehensive analysis.

        This is the main entry point for fetching all events, markets, and detailed data
        from the 7 prediction market platforms via PolyRouter.

        Parameters
        ----------
        query : Optional[str]
            Search query to filter markets (if None, fetches all)
        limit : int
            Maximum number of markets to fetch. Default: 100
        min_liquidity : float
            Minimum liquidity filter (USD). Default: $1000
        include_orderbooks : bool
            Include detailed orderbook data (slower). Default: False
        include_sports : bool
            Include sports betting markets. Default: False

        Returns
        -------
        Dict[str, Any]
            Comprehensive market data with:
            - events: List of events from all providers
            - markets: List of markets with full data
            - market_quality_scores: Quality assessment for each market
            - provider_comparison: Analysis across providers
            - arbitrage_opportunities: All detected arbitrage types
            - recommendations: Ranked trading recommendations
            - summary_stats: Aggregate statistics
        """
        logger.info(f"Fetching complete market data (query='{query}', limit={limit}, min_liquidity=${min_liquidity:,.0f})")

        # Step 1: Fetch all markets
        if query:
            markets = await self.polyrouter.search_markets(query=query, limit=limit)
        else:
            markets = await self.polyrouter.list_markets(limit=limit)

        logger.info(f"Fetched {len(markets)} raw markets")

        # Step 2: Filter by liquidity
        liquid_markets = [
            m for m in markets
            if (m.get("liquidity") or 0) >= min_liquidity
        ]

        logger.info(f"Filtered to {len(liquid_markets)} markets with ≥${min_liquidity:,.0f} liquidity")

        # Step 3: Enrich with detailed data (if requested)
        if include_orderbooks:
            enriched_markets = []
            for market in liquid_markets:
                try:
                    market_id = market.get("id")
                    if market_id:
                        details = await self.polyrouter.get_market_details(market_id)
                        market["detailed_data"] = details
                except Exception as e:
                    logger.warning(f"Failed to get orderbook for {market.get('id')}: {e}")
                enriched_markets.append(market)
            liquid_markets = enriched_markets

        # Step 4: Score market quality
        market_quality_scores = score_markets_batch(liquid_markets)

        # Step 5: Compare providers
        provider_comparison = compare_providers(liquid_markets)

        # Step 6: Detect all arbitrage types
        logger.info("Detecting arbitrage opportunities...")

        cross_platform_arb = find_cross_platform_arbitrage(
            liquid_markets,
            threshold=0.01,
            min_liquidity=min_liquidity,
        )

        single_platform_mispricing = find_single_platform_mispricing(
            liquid_markets,
            threshold=0.05,
            min_liquidity=min_liquidity,
        )

        # Combine all opportunities
        all_opportunities = cross_platform_arb + single_platform_mispricing

        # Rank opportunities
        bankroll = 10000.0  # Default bankroll for Kelly sizing
        ranked_opportunities = rank_opportunities(all_opportunities, bankroll)

        logger.info(f"Found {len(all_opportunities)} total arbitrage opportunities")

        # Step 7: Generate recommendations
        recommendations = generate_recommendations(
            ranked_opportunities,
            market_quality_scores,
            max_recommendations=10,
        )

        # Step 8: Fetch sports data (if requested)
        sports_data = {}
        if include_sports:
            logger.info("Fetching sports betting data...")
            try:
                sports_client = SportsBettingClient(api_key=self.polyrouter.api_key)
                sports_events = await sports_client.get_all_sports_events(limit=50)
                sports_arbitrage = await sports_client.find_sports_arbitrage(
                    events=sports_events,
                    min_edge=0.02,
                )
                sports_data = {
                    "events": sports_events,
                    "arbitrage_opportunities": sports_arbitrage,
                }
                logger.info(f"Fetched {len(sports_events)} sports events, {len(sports_arbitrage)} arbitrage opportunities")
            except Exception as e:
                logger.error(f"Failed to fetch sports data: {e}")
                sports_data = {"error": str(e)}

        # Step 9: Calculate summary statistics
        total_liquidity = sum(m.get("liquidity", 0) for m in liquid_markets)
        total_volume = sum(m.get("volume_total") or m.get("volume", 0) for m in liquid_markets)
        avg_quality = sum(s.overall_score for s in market_quality_scores) / len(market_quality_scores) if market_quality_scores else 0

        summary_stats = {
            "total_markets": len(liquid_markets),
            "providers": list(provider_comparison.providers),
            "provider_count": len(provider_comparison.providers),
            "total_liquidity": total_liquidity,
            "total_volume_24h": total_volume,
            "avg_quality_score": avg_quality,
            "arbitrage_opportunity_count": len(all_opportunities),
            "high_quality_markets": len([s for s in market_quality_scores if s.overall_score >= 0.7]),
            "recommendation_count": len(recommendations),
        }

        logger.info(
            f"Complete data summary: {summary_stats['total_markets']} markets, "
            f"{summary_stats['provider_count']} providers, "
            f"{summary_stats['arbitrage_opportunity_count']} arbitrage opportunities"
        )

        # Return comprehensive data package
        return {
            "markets": liquid_markets,
            "market_quality_scores": [s.to_dict() for s in market_quality_scores],
            "provider_comparison": provider_comparison.to_dict(),
            "arbitrage_opportunities": {
                "cross_platform": [opp.to_dict() for opp in cross_platform_arb],
                "single_platform": [opp.to_dict() for opp in single_platform_mispricing],
                "all": [opp.to_dict() for opp in all_opportunities],
                "ranked": [opp.to_dict() for opp in ranked_opportunities],
            },
            "recommendations": recommendations,
            "sports_data": sports_data,
            "summary_stats": summary_stats,
            "metadata": {
                "query": query,
                "limit": limit,
                "min_liquidity": min_liquidity,
                "include_orderbooks": include_orderbooks,
                "include_sports": include_sports,
            },
        }

    async def get_sports_opportunities(
        self,
        sport: Optional[str] = None,
        league: Optional[str] = None,
        min_edge: float = 0.02,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get sports betting opportunities across all bookmakers.

        Parameters
        ----------
        sport : Optional[str]
            Filter by sport (e.g., "NFL", "NBA", "MLB")
        league : Optional[str]
            Filter by league
        min_edge : float
            Minimum arbitrage edge required. Default: 2%
        limit : int
            Maximum events to fetch. Default: 100

        Returns
        -------
        Dict[str, Any]
            Sports opportunities with:
            - events: List of sports events
            - arbitrage_opportunities: Detected arbitrage across bookmakers
            - ev_opportunities: Positive EV bets
            - player_props: Player proposition markets
            - futures: Season-long futures markets
        """
        logger.info(f"Fetching sports opportunities (sport={sport}, league={league}, min_edge={min_edge:.1%})")

        try:
            sports_client = SportsBettingClient(api_key=self.polyrouter.api_key)

            # Fetch events
            events = await sports_client.get_all_sports_events(
                sport=sport,
                league=league,
                limit=limit,
            )

            # Detect arbitrage
            arbitrage_opportunities = await sports_client.find_sports_arbitrage(
                events=events,
                min_edge=min_edge,
            )

            # Get +EV opportunities
            ev_opportunities = await sports_client.get_ev_opportunities(
                events=events,
                min_ev_percent=5.0,
            )

            # Get player props
            player_props = await sports_client.get_player_props(
                sport=sport,
                limit=50,
            )

            # Get futures
            futures = await sports_client.get_futures_markets(
                sport=sport,
                limit=50,
            )

            logger.info(
                f"Sports summary: {len(events)} events, "
                f"{len(arbitrage_opportunities)} arbitrage, "
                f"{len(ev_opportunities)} +EV bets"
            )

            return {
                "events": events,
                "arbitrage_opportunities": arbitrage_opportunities,
                "ev_opportunities": ev_opportunities,
                "player_props": player_props,
                "futures": futures,
                "summary": {
                    "event_count": len(events),
                    "arbitrage_count": len(arbitrage_opportunities),
                    "ev_count": len(ev_opportunities),
                    "prop_count": len(player_props),
                    "futures_count": len(futures),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get sports opportunities: {e}")
            return {"error": str(e)}
