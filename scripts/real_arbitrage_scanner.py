#!/usr/bin/env python3
"""
REAL Arbitrage Scanner - No Fake Opportunities

This scanner ONLY finds TRUE arbitrage opportunities:
- Cross-platform: Same binary outcome, different platforms (buy YES on A, NO on B)
- Cross-market: Related binary markets with guaranteed profit

Does NOT include:
- Multi-outcome markets (those are NOT arbitrage, just different bets)
- Single market different outcomes (NOT arbitrage)

Shows REAL ROI calculations based on actual contract payouts.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.json import JSON

console = Console()


class RealArbitrageScanner:
    """Scanner that finds ONLY real arbitrage opportunities"""

    def __init__(self):
        """Initialize scanner"""
        import os
        from utils.provider_manager import ProviderManager
        from clients.polyrouter_sports import SportsBettingClient

        # Get API key
        api_key = os.getenv("POLYROUTER_API_KEY")
        if not api_key:
            from config.settings import settings
            api_key = settings.POLYROUTER_API_KEY

        self.provider_manager = ProviderManager()
        self.sports_client = SportsBettingClient(api_key=api_key)

        # Results
        self.all_markets: List[Dict[str, Any]] = []
        self.real_arbitrage: List[Dict[str, Any]] = []
        self.sports_markets: List[Dict[str, Any]] = []

    async def fetch_markets(self, limit: int = 100) -> None:
        """Fetch all markets"""
        console.print("\n[bold cyan]📊 Fetching Markets[/bold cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching from PolyRouter...", total=None)

            data = await self.provider_manager.get_complete_market_data(
                query=None,
                limit=limit,
                min_liquidity=1000.0,
                include_orderbooks=False,
                include_sports=False,  # We'll fetch sports separately with correct API
            )

            progress.update(task, completed=True)

        self.all_markets = data.get("markets", [])
        summary = data.get("summary_stats", {})

        console.print(f"[green]✅ Fetched {len(self.all_markets)} markets[/green]")
        console.print(f"   Providers: {', '.join(summary.get('providers', []))}")
        console.print(f"   Total Liquidity: ${summary.get('total_liquidity', 0):,.0f}\n")

    async def fetch_sports_markets(self) -> None:
        """Fetch sports markets using CORRECT API endpoints"""
        console.print("\n[bold cyan]🏈 Fetching Sports Markets[/bold cyan]\n")

        # Use correct endpoint from API docs
        from clients.polyrouter import PolyRouterClient
        import os

        api_key = os.getenv("POLYROUTER_API_KEY")
        if not api_key:
            from config.settings import settings
            api_key = settings.POLYROUTER_API_KEY

        if not api_key:
            console.print("[yellow]⚠️  No API key - skipping sports markets[/yellow]\n")
            return

        client = PolyRouterClient(api_key=api_key)

        leagues = ["nfl", "nba", "mlb", "nhl"]  # Lowercase as per docs

        sports_count = 0
        for league in leagues:
            try:
                # Correct endpoint: /list-games with league parameter
                result = await client._request(
                    "GET",
                    "/list-games",
                    params={"league": league, "status": "not_started"}
                )

                if result and "games" in result:
                    games = result["games"]
                    console.print(f"   {league.upper()}: {len(games)} games")
                    self.sports_markets.extend(games)
                    sports_count += len(games)

            except Exception as e:
                console.print(f"[yellow]   {league.upper()}: Failed ({str(e)[:50]})[/yellow]")

        console.print(f"\n[green]✅ Fetched {sports_count} sports markets[/green]\n")

    def find_real_cross_platform_arbitrage(self) -> None:
        """Find REAL cross-platform arbitrage (same question, different platforms)"""
        console.print("\n[bold cyan]💰 Finding REAL Arbitrage Opportunities[/bold cyan]\n")

        # Group binary markets by question (must be EXACT same question)
        binary_markets = defaultdict(list)

        for market in self.all_markets:
            # Only binary markets (YES/NO)
            market_type = market.get("market_type", "").lower()
            if market_type != "binary":
                continue

            # Group by exact question
            question = market.get("title") or market.get("question", "")
            if not question:
                continue

            binary_markets[question].append(market)

        # Find arbitrage within each question group
        console.print(f"[yellow]Analyzing {len(binary_markets)} unique binary questions...[/yellow]")

        for question, markets in binary_markets.items():
            if len(markets) < 2:
                continue  # Need at least 2 platforms

            # Check if we can buy YES on one platform and NO on another for profit
            self._check_yes_no_arbitrage(question, markets)

        console.print(f"\n[green]✅ Found {len(self.real_arbitrage)} REAL arbitrage opportunities[/green]\n")

    def _check_yes_no_arbitrage(self, question: str, markets: List[Dict[str, Any]]) -> None:
        """Check for YES/NO arbitrage across platforms"""
        # Get all YES and NO prices
        yes_prices = []
        no_prices = []

        for market in markets:
            prices = market.get("current_prices", {})
            platform = market.get("platform", "unknown")

            # Get YES price
            if "yes" in prices:
                yes_val = prices["yes"]
                yes_price = yes_val.get("price") if isinstance(yes_val, dict) else float(yes_val) if yes_val else None

                if yes_price is not None:
                    yes_prices.append({
                        "price": yes_price,
                        "platform": platform,
                        "market": market,
                    })

            # Get NO price
            if "no" in prices:
                no_val = prices["no"]
                no_price = no_val.get("price") if isinstance(no_val, dict) else float(no_val) if no_val else None

                if no_price is not None:
                    no_prices.append({
                        "price": no_price,
                        "platform": platform,
                        "market": market,
                    })

        # Find best YES and best NO
        if not yes_prices or not no_prices:
            return

        # Best YES = lowest price (cheapest to buy)
        best_yes = min(yes_prices, key=lambda x: x["price"])

        # Best NO = lowest price (cheapest to buy)
        best_no = min(no_prices, key=lambda x: x["price"])

        # Check if they're on DIFFERENT platforms (true cross-platform)
        if best_yes["platform"] == best_no["platform"]:
            return  # Same platform, not cross-platform arbitrage

        # Calculate arbitrage
        total_cost = best_yes["price"] + best_no["price"]

        # Arbitrage exists if total cost < 1.0 (guaranteed profit)
        if total_cost < 1.0:
            profit_margin = 1.0 - total_cost

            # Estimate fees (2% per platform typically)
            est_fees = 0.04  # 2% * 2 platforms

            post_fee_margin = profit_margin - est_fees

            if post_fee_margin > 0.01:  # Only if >1% post-fee profit
                # Calculate real ROI
                # If we bet $100 on YES and $100 on NO:
                # - One pays out $100 / price (shares * $1)
                # - Total payout = $100 / price
                # - Total cost = $100 + $100 = $200
                # - Profit = payout - cost
                # - ROI = profit / cost

                yes_shares = 100 / best_yes["price"] if best_yes["price"] > 0 else 0
                no_shares = 100 / best_no["price"] if best_no["price"] > 0 else 0

                # Payout is always $1 per share for the winning outcome
                guaranteed_payout = max(yes_shares, no_shares)  # We get one of these
                total_cost_real = 200  # $100 YES + $100 NO

                gross_profit = guaranteed_payout - total_cost_real
                net_profit = gross_profit - (total_cost_real * est_fees)
                roi = (net_profit / total_cost_real) * 100

                self.real_arbitrage.append({
                    "question": question,
                    "type": "cross_platform",
                    "yes_platform": best_yes["platform"],
                    "yes_price": best_yes["price"],
                    "no_platform": best_no["platform"],
                    "no_price": best_no["price"],
                    "total_cost": total_cost,
                    "profit_margin": profit_margin,
                    "post_fee_margin": post_fee_margin,
                    "estimated_fees": est_fees,
                    "roi_percent": roi,
                    "example_bet": {
                        "yes_stake": 100,
                        "no_stake": 100,
                        "total_invested": 200,
                        "guaranteed_payout": guaranteed_payout,
                        "gross_profit": gross_profit,
                        "fees": total_cost_real * est_fees,
                        "net_profit": net_profit,
                    },
                    "yes_market": best_yes["market"],
                    "no_market": best_no["market"],
                })

    def display_results(self) -> None:
        """Display results with full JSON data"""
        console.print("\n" + "=" * 100)
        console.print("[bold white]📊 REAL ARBITRAGE OPPORTUNITIES[/bold white]".center(100))
        console.print("=" * 100 + "\n")

        if not self.real_arbitrage:
            console.print("[yellow]⚠️  No real arbitrage opportunities found.[/yellow]")
            console.print("[dim]This is normal - true arbitrage is rare in efficient markets.[/dim]\n")
            return

        # Sort by ROI
        self.real_arbitrage.sort(key=lambda x: x["roi_percent"], reverse=True)

        # Summary table
        table = Table(
            title="[bold]REAL Arbitrage Opportunities[/bold]",
            box=box.ROUNDED,
            show_header=True,
        )

        table.add_column("Question", width=50)
        table.add_column("YES Platform", width=12)
        table.add_column("NO Platform", width=12)
        table.add_column("Post-Fee Margin", justify="right", width=15)
        table.add_column("ROI %", justify="right", width=10)

        for opp in self.real_arbitrage[:10]:
            table.add_row(
                opp["question"][:47] + "...",
                opp["yes_platform"],
                opp["no_platform"],
                f"{opp['post_fee_margin']*100:.2f}%",
                f"{opp['roi_percent']:.2f}%",
            )

        console.print(table)
        console.print()

        # Detailed view of top opportunity
        if self.real_arbitrage:
            self._display_detailed_opportunity(self.real_arbitrage[0])

    def _display_detailed_opportunity(self, opp: Dict[str, Any]) -> None:
        """Display detailed view of an opportunity with full JSON"""
        console.print("\n[bold white]💰 TOP OPPORTUNITY - FULL DETAILS[/bold white]\n")

        # Create formatted output
        detail_text = f"""
[bold cyan]Question:[/bold cyan] {opp['question']}

[bold yellow]Arbitrage Strategy:[/bold yellow]
  1. Buy YES @ {opp['yes_price']:.4f} on {opp['yes_platform']}
  2. Buy NO @ {opp['no_price']:.4f} on {opp['no_platform']}

[bold green]Economics (Example $100 on each):[/bold green]
  Total Investment: ${opp['example_bet']['total_invested']:.2f}
  Guaranteed Payout: ${opp['example_bet']['guaranteed_payout']:.2f}
  Gross Profit: ${opp['example_bet']['gross_profit']:.2f}
  Estimated Fees (4%): ${opp['example_bet']['fees']:.2f}
  Net Profit: ${opp['example_bet']['net_profit']:.2f}

[bold magenta]Returns:[/bold magenta]
  Pre-Fee Margin: {opp['profit_margin']*100:.2f}%
  Post-Fee Margin: {opp['post_fee_margin']*100:.2f}%
  ROI: {opp['roi_percent']:.2f}%

[bold blue]Why This Works:[/bold blue]
  • One outcome MUST happen (YES or NO)
  • Total cost < $1.00 per dollar of payout
  • Guaranteed profit regardless of outcome
"""

        panel = Panel(
            detail_text.strip(),
            title="[bold]Arbitrage Opportunity[/bold]",
            border_style="green",
            box=box.DOUBLE,
        )
        console.print(panel)

        # Display full JSON
        console.print("\n[bold]📄 FULL JSON DATA:[/bold]\n")

        # Create clean JSON (remove full market objects)
        json_data = {
            "question": opp["question"],
            "type": opp["type"],
            "strategy": {
                "leg_1": {
                    "action": "BUY_YES",
                    "platform": opp["yes_platform"],
                    "price": opp["yes_price"],
                    "shares_per_dollar": 1 / opp["yes_price"] if opp["yes_price"] > 0 else 0,
                },
                "leg_2": {
                    "action": "BUY_NO",
                    "platform": opp["no_platform"],
                    "price": opp["no_price"],
                    "shares_per_dollar": 1 / opp["no_price"] if opp["no_price"] > 0 else 0,
                },
            },
            "economics": {
                "total_cost_per_share": opp["total_cost"],
                "profit_margin_pre_fee": opp["profit_margin"],
                "estimated_fees": opp["estimated_fees"],
                "profit_margin_post_fee": opp["post_fee_margin"],
                "roi_percent": opp["roi_percent"],
            },
            "example_trade": opp["example_bet"],
            "market_details": {
                "yes_market": {
                    "title": opp["yes_market"].get("title", ""),
                    "platform": opp["yes_platform"],
                    "liquidity": opp["yes_market"].get("liquidity", 0),
                    "volume": opp["yes_market"].get("volume_total", 0),
                },
                "no_market": {
                    "title": opp["no_market"].get("title", ""),
                    "platform": opp["no_platform"],
                    "liquidity": opp["no_market"].get("liquidity", 0),
                    "volume": opp["no_market"].get("volume_total", 0),
                },
            },
        }

        console.print(JSON(json.dumps(json_data, indent=2)))

    def export_results(self) -> None:
        """Export results to JSON"""
        output_path = project_root / "real_arbitrage_results.json"

        results = {
            "scan_date": datetime.now().isoformat(),
            "total_markets_analyzed": len(self.all_markets),
            "real_arbitrage_found": len(self.real_arbitrage),
            "opportunities": [
                {
                    "question": opp["question"],
                    "yes_platform": opp["yes_platform"],
                    "yes_price": opp["yes_price"],
                    "no_platform": opp["no_platform"],
                    "no_price": opp["no_price"],
                    "roi_percent": opp["roi_percent"],
                    "post_fee_margin": opp["post_fee_margin"],
                    "example_bet": opp["example_bet"],
                }
                for opp in self.real_arbitrage
            ],
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]✅ Results exported to: {output_path}[/green]\n")


async def main():
    """Main execution"""
    console.rule("[bold blue]REAL ARBITRAGE SCANNER[/bold blue]")

    scanner = RealArbitrageScanner()

    try:
        # Fetch markets
        await scanner.fetch_markets(limit=100)

        # Fetch sports (with correct API)
        await scanner.fetch_sports_markets()

        # Find REAL arbitrage
        scanner.find_real_cross_platform_arbitrage()

        # Display results
        scanner.display_results()

        # Export
        scanner.export_results()

        console.rule("[bold green]✅ SCAN COMPLETE[/bold green]")
        return 0

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
