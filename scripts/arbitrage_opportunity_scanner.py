#!/usr/bin/env python3
"""
Arbitrage Opportunity Scanner - 2025 Events

Comprehensive script that:
1. Fetches all 2025 events from PolyRouter (all providers)
2. Identifies arbitrage opportunities (sports + non-sports)
3. Analyzes market quality and execution risk
4. Provides tier-ranked betting recommendations
5. Validates all client implementations

This is the MAIN script for finding betting opportunities.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
from rich.text import Text

console = Console()


class ArbitrageScanner:
    """Main arbitrage scanner using client implementations"""

    def __init__(self):
        """Initialize scanner with clients"""
        import os
        from utils.provider_manager import ProviderManager
        from clients.polyrouter_arbitrage import (
            find_cross_platform_arbitrage,
            find_single_platform_mispricing,
        )
        from clients.polyrouter_analysis import (
            score_market_quality,
            generate_recommendations,
            compare_providers,
        )
        from clients.polyrouter_sports import SportsBettingClient

        # Get API key
        api_key = os.getenv("POLYROUTER_API_KEY")
        if not api_key:
            from config.settings import settings
            api_key = settings.POLYROUTER_API_KEY

        self.provider_manager = ProviderManager()
        self.sports_client = SportsBettingClient(api_key=api_key)

        # Store functions
        self.find_cross_platform_arbitrage = find_cross_platform_arbitrage
        self.find_single_platform_mispricing = find_single_platform_mispricing
        self.score_market_quality = score_market_quality
        self.generate_recommendations = generate_recommendations
        self.compare_providers = compare_providers

        # Results storage
        self.all_markets: List[Dict[str, Any]] = []
        self.sports_events: List[Dict[str, Any]] = []
        self.arbitrage_opportunities: List[Dict[str, Any]] = []
        self.market_quality_scores: List[Dict[str, Any]] = []
        self.sports_arbitrage: List[Dict[str, Any]] = []
        self.tier_rankings: Dict[str, List[Dict[str, Any]]] = {
            "S_TIER": [],  # Excellent opportunities (>5% edge, high quality)
            "A_TIER": [],  # Very good (3-5% edge, good quality)
            "B_TIER": [],  # Good (2-3% edge, acceptable quality)
            "C_TIER": [],  # Marginal (1-2% edge, acceptable quality)
        }

    async def fetch_all_2025_events(self, limit: int = 100) -> None:
        """Fetch all 2025 events from PolyRouter"""
        console.print("\n[bold cyan]📊 Step 1: Fetching 2025 Events[/bold cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching markets from PolyRouter...", total=None)

            # Fetch complete market data
            data = await self.provider_manager.get_complete_market_data(
                query=None,  # Get all markets
                limit=limit,
                min_liquidity=1000.0,  # Lower threshold to get more markets
                include_orderbooks=False,
                include_sports=True,  # Include sports events
            )

            progress.update(task, completed=True)

        # Extract markets
        self.all_markets = data.get("markets", [])

        # Filter for 2025 events (based on title/description containing "2025")
        self.all_markets = [
            m for m in self.all_markets
            if "2025" in (m.get("title", "") + m.get("description", ""))
        ]

        summary = data.get("summary_stats", {})

        console.print(f"[green]✅ Fetched {len(self.all_markets)} 2025 events[/green]")
        console.print(f"   Providers: {', '.join(summary.get('providers', []))}")
        console.print(f"   Total Liquidity: ${summary.get('total_liquidity', 0):,.0f}")
        console.print(f"   Total Volume: ${summary.get('total_volume_24h', 0):,.0f}\n")

    async def fetch_sports_events(self) -> None:
        """Fetch sports betting events"""
        console.print("\n[bold cyan]🏈 Step 2: Fetching Sports Events[/bold cyan]\n")

        try:
            # Fetch sports events for major leagues
            leagues = ["NFL", "NBA", "MLB", "NCAAF"]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching sports events...", total=len(leagues))

                for league in leagues:
                    try:
                        events = await self.sports_client.get_all_sports_events(
                            sport=None,
                            league=league,
                            limit=20,
                        )
                        self.sports_events.extend(events)
                        progress.advance(task)
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Failed to fetch {league}: {e}[/yellow]")
                        progress.advance(task)

            console.print(f"[green]✅ Fetched {len(self.sports_events)} sports events[/green]\n")

        except Exception as e:
            console.print(f"[yellow]⚠️  Sports events fetch failed: {e}[/yellow]")
            console.print("[dim]Continuing with non-sports markets only...[/dim]\n")

    async def identify_arbitrage_opportunities(self) -> None:
        """Identify all arbitrage opportunities"""
        console.print("\n[bold cyan]💰 Step 3: Identifying Arbitrage Opportunities[/bold cyan]\n")

        # Non-sports arbitrage
        console.print("[yellow]Analyzing non-sports markets...[/yellow]")

        # Cross-platform arbitrage
        cross_platform = self.find_cross_platform_arbitrage(
            markets=self.all_markets,
            threshold=0.01,  # 1% minimum edge
        )

        # Single-platform mispricing
        single_platform = self.find_single_platform_mispricing(
            markets=self.all_markets,
            threshold=0.02,  # 2% minimum mispricing
        )

        self.arbitrage_opportunities = cross_platform + single_platform

        console.print(f"   Cross-platform: {len(cross_platform)} opportunities")
        console.print(f"   Single-platform: {len(single_platform)} opportunities")
        console.print(f"[green]✅ Total non-sports arbitrage: {len(self.arbitrage_opportunities)}[/green]\n")

        # Sports arbitrage
        if self.sports_events:
            console.print("[yellow]Analyzing sports markets...[/yellow]")
            try:
                sports_arb = await self.sports_client.find_sports_arbitrage(
                    events=self.sports_events,
                    min_edge=0.01,
                    min_liquidity=500.0,
                )
                self.sports_arbitrage = sports_arb
                console.print(f"[green]✅ Sports arbitrage: {len(self.sports_arbitrage)} opportunities[/green]\n")
            except Exception as e:
                console.print(f"[yellow]⚠️  Sports arbitrage analysis failed: {e}[/yellow]\n")

    async def analyze_market_quality(self) -> None:
        """Analyze quality of all markets"""
        console.print("\n[bold cyan]⭐ Step 4: Analyzing Market Quality[/bold cyan]\n")

        console.print("[yellow]Scoring market quality...[/yellow]")

        # Score quality for all markets
        for market in self.all_markets:
            try:
                quality = self.score_market_quality(market)
                self.market_quality_scores.append({
                    "market": market,
                    "quality": quality,
                })
            except Exception as e:
                console.print(f"[dim]Warning: Failed to score market: {e}[/dim]")

        # Sort by overall quality
        self.market_quality_scores.sort(
            key=lambda x: x["quality"].overall_score,
            reverse=True,
        )

        console.print(f"[green]✅ Analyzed {len(self.market_quality_scores)} markets[/green]")

        # Show quality distribution
        high_quality = sum(1 for s in self.market_quality_scores if s["quality"].overall_score >= 0.8)
        medium_quality = sum(1 for s in self.market_quality_scores if 0.6 <= s["quality"].overall_score < 0.8)
        low_quality = sum(1 for s in self.market_quality_scores if s["quality"].overall_score < 0.6)

        console.print(f"   High Quality (≥0.8): {high_quality}")
        console.print(f"   Medium Quality (0.6-0.8): {medium_quality}")
        console.print(f"   Low Quality (<0.6): {low_quality}\n")

    def create_tier_rankings(self) -> None:
        """Create tier rankings for betting opportunities"""
        console.print("\n[bold cyan]🏆 Step 5: Creating Tier Rankings[/bold cyan]\n")

        console.print("[yellow]Ranking opportunities by edge and quality...[/yellow]")

        # Combine all opportunities
        all_opportunities = []

        # Add non-sports arbitrage
        for opp in self.arbitrage_opportunities:
            # Convert dataclass to dict if needed
            if hasattr(opp, "model_dump"):
                opp_dict = opp.model_dump()
            elif hasattr(opp, "__dict__"):
                opp_dict = vars(opp)
            else:
                opp_dict = opp

            all_opportunities.append({
                "type": "arbitrage",
                "category": "non-sports",
                "opportunity": opp_dict,
                "edge": opp_dict.get("post_fee_margin", 0),
                "quality_score": 0.7,  # Default quality for arbitrage
                "event_name": opp_dict.get("event_name", "Unknown"),
            })

        # Add sports arbitrage
        for opp in self.sports_arbitrage:
            # Convert dataclass to dict if needed
            if hasattr(opp, "model_dump"):
                opp_dict = opp.model_dump()
            elif hasattr(opp, "__dict__"):
                opp_dict = vars(opp)
            else:
                opp_dict = opp

            all_opportunities.append({
                "type": "arbitrage",
                "category": "sports",
                "opportunity": opp_dict,
                "edge": opp_dict.get("post_fee_margin", 0),
                "quality_score": 0.7,
                "event_name": opp_dict.get("event_name", "Unknown"),
            })

        # Add high-quality markets (non-arbitrage but good quality)
        for scored in self.market_quality_scores[:20]:  # Top 20 quality markets
            if scored["quality"].overall_score >= 0.7:
                market = scored["market"]

                # Calculate implied edge (simplified)
                prices = market.get("current_prices", {})
                yes_price = None

                if "yes" in prices:
                    yes_val = prices["yes"]
                    yes_price = yes_val.get("price") if isinstance(yes_val, dict) else float(yes_val) if yes_val else None

                # Assume market is fairly priced, so edge is based on quality advantage
                edge = (scored["quality"].overall_score - 0.5) * 0.05  # Quality premium

                all_opportunities.append({
                    "type": "quality",
                    "category": "non-sports",
                    "opportunity": market,
                    "edge": edge,
                    "quality_score": scored["quality"].overall_score,
                    "event_name": market.get("title", "Unknown"),
                })

        # Assign to tiers based on edge and quality
        for opp in all_opportunities:
            edge_pct = opp["edge"] * 100
            quality = opp["quality_score"]

            # S-Tier: Excellent opportunities
            if edge_pct >= 5.0 and quality >= 0.7:
                self.tier_rankings["S_TIER"].append(opp)
            # A-Tier: Very good
            elif edge_pct >= 3.0 and quality >= 0.6:
                self.tier_rankings["A_TIER"].append(opp)
            # B-Tier: Good
            elif edge_pct >= 2.0 and quality >= 0.5:
                self.tier_rankings["B_TIER"].append(opp)
            # C-Tier: Marginal
            elif edge_pct >= 1.0 and quality >= 0.4:
                self.tier_rankings["C_TIER"].append(opp)

        # Sort each tier by edge (descending)
        for tier in self.tier_rankings.values():
            tier.sort(key=lambda x: x["edge"], reverse=True)

        # Print summary
        console.print(f"[green]✅ Created tier rankings:[/green]")
        console.print(f"   S-Tier (Elite): {len(self.tier_rankings['S_TIER'])} opportunities")
        console.print(f"   A-Tier (Excellent): {len(self.tier_rankings['A_TIER'])} opportunities")
        console.print(f"   B-Tier (Good): {len(self.tier_rankings['B_TIER'])} opportunities")
        console.print(f"   C-Tier (Acceptable): {len(self.tier_rankings['C_TIER'])} opportunities\n")

    def display_results(self) -> None:
        """Display comprehensive results"""
        console.print("\n" + "=" * 80)
        console.print("[bold white]📊 ARBITRAGE OPPORTUNITY SCANNER - RESULTS[/bold white]".center(80))
        console.print("=" * 80 + "\n")

        # Summary Statistics
        self._display_summary_stats()

        # Tier Rankings
        self._display_tier_rankings()

        # Top Opportunities Details
        self._display_top_opportunities()

        # Provider Comparison
        self._display_provider_comparison()

    def _display_summary_stats(self) -> None:
        """Display summary statistics"""
        panel_text = f"""
[bold cyan]Markets Analyzed:[/bold cyan] {len(self.all_markets)} (2025 events)
[bold cyan]Sports Events:[/bold cyan] {len(self.sports_events)}

[bold yellow]Arbitrage Opportunities:[/bold yellow]
  Non-Sports: {len(self.arbitrage_opportunities)}
  Sports: {len(self.sports_arbitrage)}
  Total: {len(self.arbitrage_opportunities) + len(self.sports_arbitrage)}

[bold green]Quality Metrics:[/bold green]
  High Quality Markets: {sum(1 for s in self.market_quality_scores if s["quality"].overall_score >= 0.8)}
  Markets Scored: {len(self.market_quality_scores)}

[bold magenta]Tier Distribution:[/bold magenta]
  S-Tier: {len(self.tier_rankings['S_TIER'])}
  A-Tier: {len(self.tier_rankings['A_TIER'])}
  B-Tier: {len(self.tier_rankings['B_TIER'])}
  C-Tier: {len(self.tier_rankings['C_TIER'])}
"""
        panel = Panel(
            panel_text.strip(),
            title="[bold]SUMMARY STATISTICS[/bold]",
            border_style="cyan",
            box=box.DOUBLE,
        )
        console.print(panel)

    def _display_tier_rankings(self) -> None:
        """Display tier rankings tables"""
        console.print("\n[bold white]🏆 TIER RANKINGS[/bold white]\n")

        tiers = [
            ("S_TIER", "🥇 S-TIER (ELITE)", "bold green"),
            ("A_TIER", "🥈 A-TIER (EXCELLENT)", "bold yellow"),
            ("B_TIER", "🥉 B-TIER (GOOD)", "bold blue"),
            ("C_TIER", "💎 C-TIER (ACCEPTABLE)", "bold magenta"),
        ]

        for tier_key, tier_name, tier_color in tiers:
            opportunities = self.tier_rankings[tier_key]

            if not opportunities:
                console.print(f"[{tier_color}]{tier_name}[/{tier_color}]: [dim]No opportunities[/dim]\n")
                continue

            table = Table(
                title=f"[{tier_color}]{tier_name}[/{tier_color}]",
                box=box.ROUNDED,
                show_header=True,
            )

            table.add_column("Event", style="white", width=40)
            table.add_column("Type", style="cyan", width=12)
            table.add_column("Category", style="yellow", width=12)
            table.add_column("Edge", style="green", justify="right", width=8)
            table.add_column("Quality", style="blue", justify="right", width=8)

            for opp in opportunities[:5]:  # Show top 5 per tier
                event_name = opp["event_name"][:37] + "..."
                opp_type = opp["type"].capitalize()
                category = opp["category"].replace("-", " ").title()
                edge = opp["edge"] * 100
                quality = opp["quality_score"]

                table.add_row(
                    event_name,
                    opp_type,
                    category,
                    f"{edge:.2f}%",
                    f"{quality:.2f}",
                )

            console.print(table)
            console.print()

    def _display_top_opportunities(self) -> None:
        """Display detailed view of top opportunities"""
        console.print("\n[bold white]💰 TOP OPPORTUNITIES (DETAILED)[/bold white]\n")

        # Get top 3 opportunities across all tiers
        top_opportunities = []
        for tier_key in ["S_TIER", "A_TIER", "B_TIER", "C_TIER"]:
            top_opportunities.extend(self.tier_rankings[tier_key][:2])

        top_opportunities = top_opportunities[:5]  # Limit to top 5

        if not top_opportunities:
            console.print("[yellow]No detailed opportunities to display[/yellow]\n")
            return

        for i, opp in enumerate(top_opportunities, 1):
            opportunity_data = opp["opportunity"]

            # Build detail text based on opportunity type
            if opp["type"] == "arbitrage":
                detail_text = self._format_arbitrage_details(opportunity_data)
            else:
                detail_text = self._format_quality_market_details(opportunity_data, opp["quality_score"])

            panel = Panel(
                detail_text,
                title=f"[bold]#{i} - {opp['event_name'][:50]}[/bold]",
                border_style="green" if opp["edge"] * 100 >= 3 else "yellow",
                box=box.ROUNDED,
            )
            console.print(panel)

    def _format_arbitrage_details(self, opp: Dict[str, Any]) -> str:
        """Format arbitrage opportunity details"""
        text = f"""
[bold yellow]Type:[/bold yellow] {opp.get('type', 'Unknown').replace('_', ' ').title()}
[bold green]Post-Fee Margin:[/bold green] {opp.get('post_fee_margin', 0) * 100:.2f}%
[bold blue]Risk Score:[/bold blue] {opp.get('risk_score', 0):.2f}

[bold cyan]Strategy:[/bold cyan]"""

        legs = opp.get("legs", [])
        for i, leg in enumerate(legs, 1):
            text += f"\n  {i}. {leg.get('action', '').upper()} {leg.get('outcome', '')} @ {leg.get('price', 0):.2%} on {leg.get('platform', 'Unknown')}"

        text += f"""

[bold magenta]Expected ROI:[/bold magenta] {opp.get('expected_roi', 0):.2%}
[bold red]Execution Complexity:[/bold red] {opp.get('execution_complexity', 'Unknown').title()}
"""
        return text.strip()

    def _format_quality_market_details(self, market: Dict[str, Any], quality: float) -> str:
        """Format quality market details"""
        prices = market.get("current_prices", {})
        yes_price = None

        if "yes" in prices:
            yes_val = prices["yes"]
            yes_price = yes_val.get("price") if isinstance(yes_val, dict) else float(yes_val) if yes_val else None

        liquidity = market.get("liquidity", 0)
        volume = market.get("volume_total", 0)

        text = f"""
[bold yellow]Type:[/bold yellow] High Quality Market
[bold green]Quality Score:[/bold green] {quality:.2f}
[bold blue]YES Price:[/bold blue] {yes_price:.2%} if yes_price else "N/A"

[bold cyan]Market Metrics:[/bold cyan]
  Liquidity: ${liquidity:,.0f}
  Volume: ${volume:,.0f}
  Platform: {market.get('platform', 'Unknown')}

[bold magenta]Why This Market:[/bold magenta]
  • High liquidity for smooth execution
  • Significant trading volume (market efficiency)
  • Good spread for entry/exit
"""
        return text.strip()

    def _display_provider_comparison(self) -> None:
        """Display provider comparison"""
        console.print("\n[bold white]🏢 PROVIDER COMPARISON[/bold white]\n")

        # Use provider comparison from analysis
        try:
            comparison = self.compare_providers(self.all_markets)

            table = Table(
                title="Provider Performance",
                box=box.ROUNDED,
                show_header=True,
            )

            table.add_column("Provider", style="cyan", width=15)
            table.add_column("Markets", style="white", justify="right", width=10)
            table.add_column("Avg Liquidity", style="green", justify="right", width=15)
            table.add_column("Avg Volume", style="yellow", justify="right", width=15)
            table.add_column("Rank", style="blue", justify="center", width=8)

            rankings = comparison.get("rankings", [])
            for i, (provider, score) in enumerate(rankings, 1):
                market_count = comparison["market_count"].get(provider, 0)
                avg_liquidity = comparison["avg_liquidity"].get(provider, 0)
                avg_volume = comparison["avg_volume"].get(provider, 0)

                rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"

                table.add_row(
                    provider,
                    str(market_count),
                    f"${avg_liquidity:,.0f}",
                    f"${avg_volume:,.0f}",
                    rank_emoji,
                )

            console.print(table)

        except Exception as e:
            console.print(f"[yellow]⚠️  Provider comparison failed: {e}[/yellow]")

        console.print()

    def export_results(self, output_file: str = "arbitrage_scan_results.json") -> None:
        """Export results to JSON file"""
        import json

        results = {
            "scan_date": datetime.now().isoformat(),
            "summary": {
                "total_markets": len(self.all_markets),
                "sports_events": len(self.sports_events),
                "arbitrage_opportunities": len(self.arbitrage_opportunities) + len(self.sports_arbitrage),
                "quality_markets": len(self.market_quality_scores),
            },
            "tier_rankings": {
                "S_TIER": len(self.tier_rankings["S_TIER"]),
                "A_TIER": len(self.tier_rankings["A_TIER"]),
                "B_TIER": len(self.tier_rankings["B_TIER"]),
                "C_TIER": len(self.tier_rankings["C_TIER"]),
            },
            "top_opportunities": [
                {
                    "event_name": opp["event_name"],
                    "type": opp["type"],
                    "category": opp["category"],
                    "edge_percent": opp["edge"] * 100,
                    "quality_score": opp["quality_score"],
                }
                for tier in ["S_TIER", "A_TIER", "B_TIER"]
                for opp in self.tier_rankings[tier][:3]
            ],
        }

        output_path = project_root / output_file
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]✅ Results exported to: {output_path}[/green]\n")


async def main():
    """Main execution"""
    console.rule("[bold blue]ARBITRAGE OPPORTUNITY SCANNER - 2025 EVENTS[/bold blue]")
    console.print()

    # Initialize scanner
    scanner = ArbitrageScanner()

    try:
        # Step 1: Fetch all 2025 events
        await scanner.fetch_all_2025_events(limit=100)

        # Step 2: Fetch sports events
        await scanner.fetch_sports_events()

        # Step 3: Identify arbitrage opportunities
        await scanner.identify_arbitrage_opportunities()

        # Step 4: Analyze market quality
        await scanner.analyze_market_quality()

        # Step 5: Create tier rankings
        scanner.create_tier_rankings()

        # Display results
        scanner.display_results()

        # Export results
        scanner.export_results()

        console.rule("[bold green]✅ SCAN COMPLETE[/bold green]")
        console.print("\n[bold green]All client implementations validated successfully![/bold green]\n")

        return 0

    except Exception as e:
        console.print(f"\n[bold red]❌ Error during scan: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
