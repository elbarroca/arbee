"""
polyrouter_sports.py

Enhanced sports betting functionality for PolyRouter API.

This module provides specialized methods for sports betting markets including:
- League and game data retrieval
- Player props and awards markets
- Futures markets (season-long bets)
- Sports-specific arbitrage detection
- Cross-bookmaker odds comparison

Public API:
- class SportsBettingClient
    - async get_all_sports_events(...)
    - async get_player_props(...)
    - async find_sports_arbitrage(...)
    - async compare_bookmaker_odds(...)
    - async get_ev_opportunities(...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict

from clients.polyrouter import PolyRouterClient

__all__ = [
    "SportsBettingClient",
]

logger = logging.getLogger(__name__)


class SportsBettingClient:
    """Enhanced sports betting client wrapping PolyRouter with specialized methods.

    Parameters
    ----------
    api_key : str
        PolyRouter API key
    base_url : str, optional
        API base URL. Default: "https://api.polyrouter.io/functions/v1"
    timeout : float, optional
        Request timeout in seconds. Default: 30.0

    Examples
    --------
    >>> client = SportsBettingClient(api_key="your-key")
    >>> events = await client.get_all_sports_events(sport="NFL")
    >>> arb_ops = await client.find_sports_arbitrage(events)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polyrouter.io/functions/v1",
        timeout: float = 30.0,
    ) -> None:
        self.client = PolyRouterClient(api_key=api_key, base_url=base_url, timeout=timeout)
        logger.info("SportsBettingClient initialized")

    # ==================== Core Sports Data Retrieval ==================== #

    async def get_all_sports_events(
        self,
        sport: Optional[str] = None,
        league: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get all available sports events with their betting markets.

        Parameters
        ----------
        sport : Optional[str]
            Filter by sport (e.g., "NFL", "NBA", "MLB")
        league : Optional[str]
            Filter by league (e.g., "NFL", "Premier League")
        limit : int
            Maximum number of events to return

        Returns
        -------
        List[Dict[str, Any]]
            List of sports events with markets
        """
        try:
            # Get games
            games = await self.client.list_games(league=league, limit=limit)

            # Enrich each game with markets
            enriched_events = []
            for game in games:
                game_id = game.get("id")
                if game_id:
                    try:
                        markets = await self.client.get_game_markets(game_id)
                        game["markets"] = markets
                        game["market_count"] = len(markets)
                    except Exception as e:
                        logger.warning(f"Failed to get markets for game {game_id}: {e}")
                        game["markets"] = []
                        game["market_count"] = 0

                enriched_events.append(game)

            # Filter by sport if specified
            if sport:
                enriched_events = [
                    e for e in enriched_events
                    if e.get("sport", "").lower() == sport.lower()
                ]

            logger.info(f"Retrieved {len(enriched_events)} sports events")
            return enriched_events

        except Exception as e:
            logger.error(f"Failed to get sports events: {e}")
            return []

    async def get_player_props(
        self,
        player_name: Optional[str] = None,
        sport: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get player proposition betting markets.

        Parameters
        ----------
        player_name : Optional[str]
            Filter by player name (partial match)
        sport : Optional[str]
            Filter by sport
        limit : int
            Maximum number of prop markets to return

        Returns
        -------
        List[Dict[str, Any]]
            List of player prop markets
        """
        try:
            # Get all awards (includes player props)
            props = await self.client.list_awards(sport=sport, limit=limit)

            # Filter by player name if specified
            if player_name:
                props = [
                    p for p in props
                    if player_name.lower() in p.get("title", "").lower()
                    or player_name.lower() in p.get("player", "").lower()
                ]

            # Enrich with odds
            enriched_props = []
            for prop in props:
                prop_id = prop.get("id")
                if prop_id:
                    try:
                        odds = await self.client.get_award_odds(prop_id)
                        prop["odds"] = odds
                    except Exception as e:
                        logger.warning(f"Failed to get odds for prop {prop_id}: {e}")
                        prop["odds"] = {}

                enriched_props.append(prop)

            logger.info(f"Retrieved {len(enriched_props)} player prop markets")
            return enriched_props

        except Exception as e:
            logger.error(f"Failed to get player props: {e}")
            return []

    async def get_futures_markets(
        self,
        sport: Optional[str] = None,
        market_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get futures markets (season-long bets).

        Parameters
        ----------
        sport : Optional[str]
            Filter by sport
        market_type : Optional[str]
            Filter by market type (e.g., "championship", "mvp", "playoffs")
        limit : int
            Maximum number of futures to return

        Returns
        -------
        List[Dict[str, Any]]
            List of futures markets with odds
        """
        try:
            # Get futures
            futures = await self.client.list_futures(sport=sport, limit=limit)

            # Filter by market type if specified
            if market_type:
                futures = [
                    f for f in futures
                    if market_type.lower() in f.get("type", "").lower()
                ]

            # Enrich with odds
            enriched_futures = []
            for future in futures:
                future_id = future.get("id")
                if future_id:
                    try:
                        odds = await self.client.get_futures_odds(future_id)
                        future["odds"] = odds
                    except Exception as e:
                        logger.warning(f"Failed to get odds for future {future_id}: {e}")
                        future["odds"] = {}

                enriched_futures.append(future)

            logger.info(f"Retrieved {len(enriched_futures)} futures markets")
            return enriched_futures

        except Exception as e:
            logger.error(f"Failed to get futures markets: {e}")
            return []

    # ==================== Arbitrage Detection ==================== #

    async def find_sports_arbitrage(
        self,
        events: Optional[List[Dict[str, Any]]] = None,
        min_edge: float = 0.02,
        min_liquidity: float = 1000.0,
    ) -> List[Dict[str, Any]]:
        """Detect arbitrage opportunities in sports betting markets.

        Parameters
        ----------
        events : Optional[List[Dict[str, Any]]]
            List of sports events to analyze. If None, fetches all events.
        min_edge : float
            Minimum arbitrage edge (margin) required. Default: 2%
        min_liquidity : float
            Minimum liquidity required per market. Default: $1000

        Returns
        -------
        List[Dict[str, Any]]
            List of arbitrage opportunities sorted by edge
        """
        if events is None:
            events = await self.get_all_sports_events(limit=100)

        opportunities = []

        for event in events:
            markets = event.get("markets", [])

            # Group markets by outcome type
            market_groups = self._group_markets_by_outcome(markets)

            for outcome_type, outcome_markets in market_groups.items():
                # Find best odds across bookmakers
                arb = self._detect_arbitrage_in_group(
                    outcome_markets,
                    event_name=event.get("title", "Unknown"),
                    outcome_type=outcome_type,
                    min_edge=min_edge,
                    min_liquidity=min_liquidity,
                )

                if arb:
                    opportunities.append(arb)

        # Sort by edge descending
        opportunities.sort(key=lambda x: x.get("edge", 0), reverse=True)

        logger.info(f"Found {len(opportunities)} sports arbitrage opportunities")
        return opportunities

    def _group_markets_by_outcome(
        self,
        markets: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group markets by outcome type (e.g., moneyline, spread, total)."""
        groups = defaultdict(list)
        for market in markets:
            outcome_type = market.get("market_type", "unknown")
            groups[outcome_type].append(market)
        return dict(groups)

    def _detect_arbitrage_in_group(
        self,
        markets: List[Dict[str, Any]],
        event_name: str,
        outcome_type: str,
        min_edge: float,
        min_liquidity: float,
    ) -> Optional[Dict[str, Any]]:
        """Detect arbitrage in a group of markets for the same outcome."""
        # Extract odds from all bookmakers
        bookmaker_odds = []
        for market in markets:
            bookmaker = market.get("bookmaker", market.get("provider", "unknown"))
            liquidity = market.get("liquidity", 0)

            # Skip if insufficient liquidity
            if liquidity < min_liquidity:
                continue

            # Get odds for all outcomes
            outcomes = market.get("outcomes", [])
            odds = market.get("odds", {})

            bookmaker_odds.append({
                "bookmaker": bookmaker,
                "market_id": market.get("id"),
                "outcomes": outcomes,
                "odds": odds,
                "liquidity": liquidity,
            })

        if len(bookmaker_odds) < 2:
            return None

        # Find best odds for each outcome
        unique_outcomes = set()
        for bm in bookmaker_odds:
            unique_outcomes.update(bm["outcomes"])

        best_odds_per_outcome = {}
        for outcome in unique_outcomes:
            best_odds = None
            best_bookmaker = None

            for bm in bookmaker_odds:
                if outcome in bm["outcomes"]:
                    idx = bm["outcomes"].index(outcome)
                    odds_list = bm["odds"].get("decimal", [])
                    if idx < len(odds_list):
                        odds_value = odds_list[idx]
                        if best_odds is None or odds_value > best_odds:
                            best_odds = odds_value
                            best_bookmaker = bm["bookmaker"]

            if best_odds and best_bookmaker:
                best_odds_per_outcome[outcome] = {
                    "odds": best_odds,
                    "bookmaker": best_bookmaker,
                    "implied_prob": 1.0 / best_odds if best_odds > 0 else 0,
                }

        # Calculate total implied probability
        total_implied_prob = sum(
            data["implied_prob"] for data in best_odds_per_outcome.values()
        )

        # Check for arbitrage (total implied prob < 1.0)
        if total_implied_prob >= 1.0:
            return None

        edge = 1.0 - total_implied_prob
        if edge < min_edge:
            return None

        # Return arbitrage opportunity
        return {
            "event_name": event_name,
            "outcome_type": outcome_type,
            "edge": edge,
            "edge_percent": edge * 100,
            "total_implied_prob": total_implied_prob,
            "best_odds": best_odds_per_outcome,
            "num_bookmakers": len(bookmaker_odds),
        }

    # ==================== Odds Comparison ==================== #

    async def compare_bookmaker_odds(
        self,
        event_id: str,
        market_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare odds across all bookmakers for a specific event.

        Parameters
        ----------
        event_id : str
            Event ID to analyze
        market_type : Optional[str]
            Filter by market type (e.g., "moneyline", "spread")

        Returns
        -------
        Dict[str, Any]
            Comparison data with best odds per outcome
        """
        try:
            markets = await self.client.get_game_markets(event_id)

            # Filter by market type if specified
            if market_type:
                markets = [m for m in markets if m.get("market_type") == market_type]

            comparison = {
                "event_id": event_id,
                "market_type": market_type or "all",
                "bookmakers": [],
                "best_odds": {},
            }

            # Collect all odds
            all_outcomes = set()
            for market in markets:
                bookmaker = market.get("bookmaker", market.get("provider"))
                outcomes = market.get("outcomes", [])
                odds = market.get("odds", {})

                all_outcomes.update(outcomes)

                comparison["bookmakers"].append({
                    "bookmaker": bookmaker,
                    "outcomes": outcomes,
                    "odds": odds,
                })

            # Find best odds for each outcome
            for outcome in all_outcomes:
                best_odds = None
                best_bookmaker = None

                for bm_data in comparison["bookmakers"]:
                    if outcome in bm_data["outcomes"]:
                        idx = bm_data["outcomes"].index(outcome)
                        odds_list = bm_data["odds"].get("decimal", [])
                        if idx < len(odds_list):
                            odds_value = odds_list[idx]
                            if best_odds is None or odds_value > best_odds:
                                best_odds = odds_value
                                best_bookmaker = bm_data["bookmaker"]

                if best_odds:
                    comparison["best_odds"][outcome] = {
                        "odds": best_odds,
                        "bookmaker": best_bookmaker,
                    }

            logger.info(f"Compared odds across {len(comparison['bookmakers'])} bookmakers")
            return comparison

        except Exception as e:
            logger.error(f"Failed to compare bookmaker odds: {e}")
            return {}

    # ==================== EV Opportunities ==================== #

    async def get_ev_opportunities(
        self,
        events: Optional[List[Dict[str, Any]]] = None,
        min_ev_percent: float = 5.0,
        true_probability_model: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Find positive expected value (EV) betting opportunities.

        Parameters
        ----------
        events : Optional[List[Dict[str, Any]]]
            List of sports events to analyze
        min_ev_percent : float
            Minimum EV percentage required. Default: 5%
        true_probability_model : Optional[callable]
            Function to estimate true probabilities. If None, uses simple no-vig method.

        Returns
        -------
        List[Dict[str, Any]]
            List of +EV opportunities sorted by EV
        """
        if events is None:
            events = await self.get_all_sports_events(limit=100)

        opportunities = []

        for event in events:
            markets = event.get("markets", [])

            for market in markets:
                bookmaker = market.get("bookmaker", market.get("provider"))
                outcomes = market.get("outcomes", [])
                odds = market.get("odds", {}).get("decimal", [])

                if len(outcomes) != len(odds):
                    continue

                # Calculate implied probabilities
                implied_probs = [1.0 / odd if odd > 0 else 0 for odd in odds]
                total_implied = sum(implied_probs)

                # Estimate true probabilities
                if true_probability_model:
                    true_probs = true_probability_model(event, market, outcomes)
                else:
                    # Simple no-vig method
                    true_probs = [prob / total_implied for prob in implied_probs]

                # Calculate EV for each outcome
                for i, (outcome, odd, true_prob) in enumerate(zip(outcomes, odds, true_probs)):
                    payout = odd
                    ev = (true_prob * payout) - 1.0
                    ev_percent = ev * 100

                    if ev_percent >= min_ev_percent:
                        opportunities.append({
                            "event_name": event.get("title", "Unknown"),
                            "bookmaker": bookmaker,
                            "outcome": outcome,
                            "odds": odd,
                            "implied_prob": implied_probs[i],
                            "true_prob": true_prob,
                            "ev": ev,
                            "ev_percent": ev_percent,
                            "edge": true_prob - implied_probs[i],
                        })

        # Sort by EV descending
        opportunities.sort(key=lambda x: x.get("ev", 0), reverse=True)

        logger.info(f"Found {len(opportunities)} +EV opportunities")
        return opportunities
