"""
Event Fetcher Agent - Fetch and filter betting events using PolyRouter.

This agent is the first step in the betting workflow. It:
1. Fetches events from PolyRouter (aggregating all providers)
2. Filters by liquidity thresholds (min $10K)
3. Filters by volume (min $100K for arbitrage opportunities)
4. Filters by spread quality (max 200 bps)
5. Returns ranked list of tradeable events in BettingEvent schema format

Part of the new betting workflow:
Event Fetcher → Market Analyzer → Researcher → Market Pricer → Professional Bettor → Reporter
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agents.base import AutonomousReActAgent, AgentState
from config.settings import Settings
from models.schemas import BettingEvent
from tools.polyrouter_tools import (
    fetch_events_from_polyrouter_tool,
    search_markets_across_providers_tool,
)

logger = logging.getLogger(__name__)


class EventFetcherInput(BaseModel):
    """Input for Event Fetcher agent"""
    market_question: Optional[str] = Field(None, description="Optional search query for markets")
    min_liquidity: float = Field(10_000.0, ge=0.0, description="Minimum liquidity in USD")
    min_volume: float = Field(100_000.0, ge=0.0, description="Minimum 24h volume in USD")
    max_spread_bps: int = Field(200, ge=0, description="Maximum spread in basis points")
    limit: int = Field(50, ge=1, le=200, description="Maximum number of events to fetch")


class EventFetcherOutput(BaseModel):
    """Output from Event Fetcher agent"""
    events: List[BettingEvent] = Field(default_factory=list, description="Filtered betting events")
    target_markets: List[Dict[str, Any]] = Field(default_factory=list, description="Specific markets matching query")
    total_events_fetched: int = Field(default=0, description="Total events fetched before filtering")
    events_after_filter: int = Field(default=0, description="Events after applying filters")
    filter_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of filtering criteria")
    providers_used: List[str] = Field(default_factory=list, description="Providers included in search")


class EventFetcherAgent(AutonomousReActAgent):
    """
    Event Fetcher Agent - First agent in betting workflow.

    Responsibilities:
    - Fetch events from PolyRouter (multi-provider aggregation)
    - Apply quality filters (liquidity, volume, spread)
    - Return ranked list of tradeable events
    - Identify specific markets matching user query

    This agent focuses on data retrieval and filtering, not analysis.
    Market analysis happens in the next agent (Market Analyzer).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_iterations: int = 10,
        **kwargs,
    ):
        """Initialize Event Fetcher agent with conservative settings."""
        super().__init__(
            settings=settings,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            enable_auto_memory_query=False,  # No memory needed for fetching
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        """System prompt for Event Fetcher agent."""
        return """You are the Event Fetcher Agent - the first agent in the POLYSEER betting workflow.

Your SOLE responsibility is to fetch and filter betting events using PolyRouter.

## YOUR TASK:
1. Use fetch_events_from_polyrouter_tool to get events from all providers
2. If user provided a market_question, also use search_markets_across_providers_tool
3. Apply filters:
   - Minimum liquidity: $10,000
   - Minimum volume: $100,000 (for arbitrage opportunities)
   - Maximum spread: 200 bps (2%)
4. Return filtered events as BettingEvent list
5. Identify target_markets that match the user's query

## TOOLS AVAILABLE:
- fetch_events_from_polyrouter_tool: Get all events from PolyRouter
- search_markets_across_providers_tool: Search for specific markets

## IMPORTANT RULES:
- DO NOT analyze markets (that's Market Analyzer's job)
- DO NOT calculate probabilities (that's Market Pricer's job)
- DO NOT make betting decisions (that's Professional Bettor's job)
- ONLY fetch, filter, and return events
- Be efficient: fetch once, filter locally
- Log why events were filtered out

## OUTPUT FORMAT:
You MUST output a structured EventFetcherOutput with:
- events: List of BettingEvent objects that passed filters
- target_markets: Specific markets matching user query
- filter_summary: Statistics about filtering process

## COMPLETION CRITERIA:
You are done when you have:
1. Fetched events from PolyRouter
2. Applied all filters
3. Identified target markets (if query provided)
4. Populated intermediate_results with filtered data
5. Set 'task_complete' = True in intermediate_results

BE CONCISE. DO NOT over-explain. Fetch, filter, return.
"""

    def get_tools(self) -> List[BaseTool]:
        """Return tools for Event Fetcher agent."""
        return [
            fetch_events_from_polyrouter_tool,
            search_markets_across_providers_tool,
        ]

    async def is_task_complete(self, state: AgentState) -> bool:
        """Check if event fetching is complete."""
        intermediate = state.get("intermediate_results", {})

        # Task is complete if we have fetched and filtered events
        if intermediate.get("task_complete"):
            return True

        # Also complete if we have events and target_markets in intermediate
        has_events = "filtered_events" in intermediate
        has_target = "target_markets" in intermediate

        # For queries, we need both. For general fetching, just events is enough.
        task_input = state.get("task_input", {})
        has_query = bool(task_input.get("market_question"))

        if has_query:
            return has_events and has_target
        else:
            return has_events

    async def extract_final_output(self, state: AgentState) -> EventFetcherOutput:
        """Extract final output from agent state."""
        intermediate = state.get("intermediate_results", {})
        task_input = state.get("task_input", {})

        # Extract filtered events
        filtered_events = intermediate.get("filtered_events", [])
        target_markets = intermediate.get("target_markets", [])
        total_fetched = intermediate.get("total_events_fetched", 0)
        providers = intermediate.get("providers_used", [])

        # Build filter summary
        filter_summary = {
            "min_liquidity": task_input.get("min_liquidity", 10_000.0),
            "min_volume": task_input.get("min_volume", 100_000.0),
            "max_spread_bps": task_input.get("max_spread_bps", 200),
            "total_fetched": total_fetched,
            "after_liquidity_filter": intermediate.get("after_liquidity_filter", 0),
            "after_volume_filter": intermediate.get("after_volume_filter", 0),
            "after_spread_filter": intermediate.get("after_spread_filter", 0),
            "final_count": len(filtered_events),
        }

        # Convert filtered_events to BettingEvent objects
        betting_events: List[BettingEvent] = []
        for event in filtered_events:
            if isinstance(event, BettingEvent):
                betting_events.append(event)
            elif isinstance(event, dict):
                betting_events.append(BettingEvent(**event))
            else:
                logger.warning(f"Unexpected event type: {type(event)}")

        return EventFetcherOutput(
            events=betting_events,
            target_markets=target_markets,
            total_events_fetched=total_fetched,
            events_after_filter=len(betting_events),
            filter_summary=filter_summary,
            providers_used=providers,
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def filter_events_by_liquidity(
    events: List[Dict[str, Any]],
    min_liquidity: float,
) -> List[Dict[str, Any]]:
    """Filter events by minimum liquidity.

    Args:
        events: List of event dictionaries
        min_liquidity: Minimum liquidity in USD

    Returns:
        Filtered list of events
    """
    filtered = []
    for event in events:
        liquidity = event.get("total_liquidity") or event.get("liquidity") or 0.0
        if liquidity >= min_liquidity:
            filtered.append(event)
        else:
            logger.debug(f"Filtered out {event.get('title', 'Unknown')}: liquidity ${liquidity:.0f} < ${min_liquidity:.0f}")

    return filtered


def filter_events_by_volume(
    events: List[Dict[str, Any]],
    min_volume: float,
) -> List[Dict[str, Any]]:
    """Filter events by minimum 24h volume.

    Args:
        events: List of event dictionaries
        min_volume: Minimum 24h volume in USD

    Returns:
        Filtered list of events
    """
    filtered = []
    for event in events:
        volume = event.get("total_volume") or event.get("volume") or event.get("volume_total") or 0.0
        if volume >= min_volume:
            filtered.append(event)
        else:
            logger.debug(f"Filtered out {event.get('title', 'Unknown')}: volume ${volume:.0f} < ${min_volume:.0f}")

    return filtered


def filter_markets_by_spread(
    markets: List[Dict[str, Any]],
    max_spread_bps: int,
) -> List[Dict[str, Any]]:
    """Filter markets by maximum spread in basis points.

    Args:
        markets: List of market dictionaries
        max_spread_bps: Maximum spread in basis points (e.g., 200 = 2%)

    Returns:
        Filtered list of markets
    """
    filtered = []
    for market in markets:
        # Get spread from market
        spread_bps = market.get("spread_bps")

        # If spread_bps not provided, calculate from prices
        if spread_bps is None:
            prices = market.get("current_prices", {})
            yes_price = None
            no_price = None

            # Handle various price formats
            if "yes" in prices:
                yes_val = prices["yes"]
                yes_price = yes_val.get("price") if isinstance(yes_val, dict) else float(yes_val) if yes_val else None

            if "no" in prices:
                no_val = prices["no"]
                no_price = no_val.get("price") if isinstance(no_val, dict) else float(no_val) if no_val else None

            # Calculate spread if we have both prices
            if yes_price is not None and no_price is not None:
                spread = (yes_price + no_price - 1.0)
                spread_bps = int(spread * 10_000)
            else:
                # No spread data available - be conservative and include it
                spread_bps = 0

        if spread_bps <= max_spread_bps:
            filtered.append(market)
        else:
            logger.debug(f"Filtered out {market.get('title', 'Unknown')}: spread {spread_bps} bps > {max_spread_bps} bps")

    return filtered


def convert_to_betting_events(
    markets: List[Dict[str, Any]],
) -> List[BettingEvent]:
    """Convert market dictionaries to BettingEvent objects.

    Groups markets by event and aggregates data.

    Args:
        markets: List of market dictionaries from PolyRouter

    Returns:
        List of BettingEvent objects
    """
    # Group markets by event (title or slug)
    events_map: Dict[str, Dict[str, Any]] = {}

    for market in markets:
        # Use title as event key (could also use event_id if available)
        event_key = market.get("title") or market.get("question", "Unknown")

        if event_key not in events_map:
            events_map[event_key] = {
                "title": event_key,
                "slug": market.get("slug") or market.get("market_id", event_key.lower().replace(" ", "-")),
                "markets": [],
                "providers": set(),
                "total_volume": 0.0,
                "total_liquidity": 0.0,
            }

        # Add market to event
        events_map[event_key]["markets"].append(market)

        # Add provider
        provider = market.get("platform") or market.get("provider", "unknown")
        events_map[event_key]["providers"].add(provider)

        # Aggregate volume and liquidity
        volume = market.get("volume_total") or market.get("volume") or 0.0
        liquidity = market.get("liquidity") or 0.0
        events_map[event_key]["total_volume"] += volume
        events_map[event_key]["total_liquidity"] += liquidity

    # Convert to BettingEvent objects
    betting_events = []
    for event_data in events_map.values():
        # Convert providers set to list
        event_data["providers"] = list(event_data["providers"])

        betting_events.append(BettingEvent(**event_data))

    # Sort by total volume (descending)
    betting_events.sort(key=lambda e: e.total_volume, reverse=True)

    return betting_events


# ============================================================================
# STANDALONE FUNCTION FOR TESTING
# ============================================================================


async def fetch_and_filter_events(
    market_question: Optional[str] = None,
    min_liquidity: float = 10_000.0,
    min_volume: float = 100_000.0,
    max_spread_bps: int = 200,
    limit: int = 50,
    settings: Optional[Settings] = None,
) -> EventFetcherOutput:
    """
    Standalone function to fetch and filter events using the Event Fetcher agent.

    This is useful for testing and for calling from other parts of the system.

    Args:
        market_question: Optional search query for markets
        min_liquidity: Minimum liquidity in USD
        min_volume: Minimum 24h volume in USD
        max_spread_bps: Maximum spread in basis points
        limit: Maximum number of events to fetch
        settings: Optional Settings object

    Returns:
        EventFetcherOutput with filtered events

    Example:
        >>> output = await fetch_and_filter_events(
        ...     market_question="US Recession 2025",
        ...     min_liquidity=10_000,
        ...     min_volume=50_000,
        ... )
        >>> print(f"Found {len(output.events)} tradeable events")
        >>> for event in output.events[:5]:
        ...     print(f"  - {event.title} ({len(event.providers)} providers)")
    """
    agent = EventFetcherAgent(settings=settings)

    task_input = EventFetcherInput(
        market_question=market_question,
        min_liquidity=min_liquidity,
        min_volume=min_volume,
        max_spread_bps=max_spread_bps,
        limit=limit,
    )

    result = await agent.run(
        task_description="Fetch and filter betting events from PolyRouter",
        task_input=task_input.model_dump(),
    )

    return result


# ============================================================================
# CLI FOR TESTING
# ============================================================================


if __name__ == "__main__":
    import asyncio
    import sys

    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    async def main():
        """CLI for testing Event Fetcher agent"""
        # Get market question from command line
        market_question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

        if market_question:
            console.print(f"\n[cyan]🔍 Fetching events for:[/cyan] [bold]{market_question}[/bold]\n")
        else:
            console.print("\n[cyan]📊 Fetching all events...[/cyan]\n")

        try:
            # Fetch and filter events
            output = await fetch_and_filter_events(
                market_question=market_question,
                min_liquidity=10_000.0,
                min_volume=100_000.0,
                max_spread_bps=200,
                limit=50,
            )

            # Display results
            console.print(f"[bold green]✅ Success![/bold green]\n")
            console.print(f"Total events fetched: {output.total_events_fetched}")
            console.print(f"Events after filtering: {output.events_after_filter}")
            console.print(f"Providers used: {', '.join(output.providers_used)}\n")

            # Display filter summary
            console.print("[bold]Filter Summary:[/bold]")
            summary = output.filter_summary
            console.print(f"  Min Liquidity: ${summary.get('min_liquidity', 0):,.0f}")
            console.print(f"  Min Volume: ${summary.get('min_volume', 0):,.0f}")
            console.print(f"  Max Spread: {summary.get('max_spread_bps', 0)} bps")
            console.print(f"  After liquidity filter: {summary.get('after_liquidity_filter', 0)}")
            console.print(f"  After volume filter: {summary.get('after_volume_filter', 0)}")
            console.print(f"  After spread filter: {summary.get('after_spread_filter', 0)}")
            console.print(f"  Final count: {summary.get('final_count', 0)}\n")

            # Display top events
            if output.events:
                table = Table(
                    title=f"[bold]🎯 Top {min(10, len(output.events))} Events[/bold]",
                    box=box.ROUNDED,
                )
                table.add_column("Event", style="cyan", width=50)
                table.add_column("Providers", style="yellow", width=15)
                table.add_column("Volume", style="green", justify="right", width=12)
                table.add_column("Liquidity", style="blue", justify="right", width=12)

                for event in output.events[:10]:
                    table.add_row(
                        event.title[:47] + "...",
                        ", ".join(event.providers),
                        f"${event.total_volume:,.0f}",
                        f"${event.total_liquidity:,.0f}",
                    )

                console.print(table)

            # Display target markets if query was provided
            if market_question and output.target_markets:
                console.print(f"\n[bold]🎯 Target Markets ({len(output.target_markets)}):[/bold]")
                for i, market in enumerate(output.target_markets[:5], 1):
                    console.print(f"{i}. {market.get('title', 'Unknown')} on {market.get('platform', 'Unknown')}")

            console.print()

        except Exception as e:
            console.print(f"[bold red]❌ Error:[/bold red] {e}\n")
            import traceback
            traceback.print_exc()
            return 1

        return 0

    sys.exit(asyncio.run(main()))
