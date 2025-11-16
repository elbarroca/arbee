"""
Elite Trader Analysis & Copy-Trading Intelligence Engine.

Assertive execution flow:
1. Filter best traders by performance metrics
2. Fetch current open positions from elite wallets
3. Aggregate smart money consensus signals
4. Generate actionable copy-trade suggestions
5. Deliver comprehensive intelligence reports
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketGamma, PolymarketDataAPI
from clients.wallet_tracker import WalletTracker
from clients.trader_analytics import TraderAnalytics

logger = logging.getLogger(__name__)


class PolymarketTraderAnalyzer:
    """
    Elite trader analysis and copy-trading intelligence orchestrator.

    Coordinates:
    1. Best trader filtering by ROI + Win Rate
    2. Open positions surveillance for elite wallets
    3. Smart money consensus aggregation
    4. Copy-trade signal generation
    5. Tag credibility scoring
    """

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.gamma = PolymarketGamma()
        self.data_api = PolymarketDataAPI()
        self.wallet_tracker = WalletTracker(api=self.data_api, gamma=self.gamma)
        self.trader_analytics = TraderAnalytics()
        self.max_concurrent = 10

    async def _process_batches(self, items: List, process_func, batch_size: int = None) -> List:
        """Execute batch processing with assertive parallelism."""
        batch_size = batch_size or self.max_concurrent
        return [
            result
            for i in range(0, len(items), batch_size)
            for result in await asyncio.gather(*[process_func(item) for item in items[i:i + batch_size]])
        ]

    # ========================================================================
    # PHASE 1: BEST TRADER FILTERING
    # ========================================================================

    async def filter_best_traders(
        self,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60,
        min_volume: float = 10000.0,
        min_trades: int = 20,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Filter elite traders by performance metrics.

        Assertive criteria:
        - ROI >= 5% (default)
        - Win Rate >= 60% (default)
        - Volume >= $10,000
        - Trades >= 20

        Args:
            min_roi: Minimum ROI threshold (0.05 = 5%)
            min_win_rate: Minimum win rate (0.60 = 60%)
            min_volume: Minimum total volume in USDC
            min_trades: Minimum number of trades
            limit: Maximum traders to return

        Returns:
            List of elite trader profiles with stats
        """
        logger.info(f"🔍 Elite trader filtering: ROI>={min_roi*100:.0f}%, WinRate>={min_win_rate*100:.0f}%, Volume>=${min_volume:,.0f}")

        # Query wallet stats with hard filters
        query = self.db.supabase.table("wallet_stats").select("*").gte("total_volume", min_volume).gte("n_positions", min_trades).limit(limit * 2)  # Fetch extra for soft filtering

        result = query.execute()

        if not result.data:
            logger.info("📭 No wallets found matching volume and trade count criteria")
            return []

        logger.info(f"🎯 Pre-filtered {len(result.data)} wallets by volume/trades")

        # Apply ROI and win rate filters
        elite_traders = []
        for wallet in result.data:
            roi = wallet.get("roi", 0)
            win_rate = wallet.get("win_rate", 0)

            if roi >= min_roi and win_rate >= min_win_rate:
                elite_traders.append({
                    "proxy_wallet": wallet.get("proxy_wallet"),
                    "total_volume": wallet.get("total_volume", 0),
                    "realized_pnl": wallet.get("realized_pnl", 0),
                    "roi": roi,
                    "win_rate": win_rate,
                    "n_positions": wallet.get("n_positions", 0),
                    "n_wins": wallet.get("n_wins", 0),
                    "n_markets": wallet.get("n_markets", 0),
                    "n_events": wallet.get("n_events", 0),
                    "first_trade_at": wallet.get("first_trade_at"),
                    "last_trade_at": wallet.get("last_trade_at"),
                    "tier": wallet.get("tier"),
                    "is_eligible": wallet.get("is_eligible", False)
                })

        # Sort by ROI descending
        elite_traders.sort(key=lambda x: x["roi"], reverse=True)

        # Limit results
        elite_traders = elite_traders[:limit]

        logger.info(f"✅ Identified {len(elite_traders)} elite traders meeting all criteria")

        if elite_traders:
            top_roi = elite_traders[0]["roi"] * 100
            avg_roi = sum(t["roi"] for t in elite_traders) / len(elite_traders) * 100
            logger.info(f"📊 Top ROI: {top_roi:.1f}% | Average ROI: {avg_roi:.1f}%")

        return elite_traders

    async def filter_tier_a_wallets(self) -> List[str]:
        """
        Get all Tier A (top-performing) wallet addresses.

        Returns:
            List of Tier A wallet addresses
        """
        logger.info("🎯 Fetching Tier A elite wallets")

        result = self.db.supabase.table("wallet_scores").select("proxy_wallet").eq("tier", "A").execute()

        if not result.data:
            # Fallback: get from wallet_stats
            logger.info("📡 Checking wallet_stats for Tier A wallets...")
            result = self.db.supabase.table("wallet_stats").select("proxy_wallet").eq("tier", "A").execute()

        tier_a_wallets = [w.get("proxy_wallet") for w in result.data if w.get("proxy_wallet")]
        logger.info(f"✅ Found {len(tier_a_wallets)} Tier A wallets")

        return tier_a_wallets

    # PHASE 2: OPEN POSITIONS SURVEILLANCE

    async def get_best_traders_open_positions(
        self,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60,
        max_traders: int = 50
    ) -> Dict[str, Any]:
        """
        Fetch current open positions from best traders.

        What are the elite traders betting on RIGHT NOW?

        Args:
            min_roi: Minimum ROI for trader selection
            min_win_rate: Minimum win rate for trader selection
            max_traders: Maximum number of traders to analyze

        Returns:
            Comprehensive open positions intelligence
        """
        logger.info("🚀 Elite open positions surveillance initiated")

        # Get best traders
        elite_traders = await self.filter_best_traders(
            min_roi=min_roi,
            min_win_rate=min_win_rate,
            limit=max_traders
        )

        if not elite_traders:
            logger.info("📭 No elite traders found")
            return self._generate_positions_report([], [])

        wallet_addresses = [t["proxy_wallet"] for t in elite_traders]
        logger.info(f"🎯 Surveilling {len(wallet_addresses)} elite wallets")

        # Fetch open positions for all wallets
        batch_result = await self.wallet_tracker.sync_wallets_open_positions_batch(
            proxy_wallets=wallet_addresses,
            save_position=self._save_open_position,
            max_concurrency=self.max_concurrent,
            progress_callback=self._log_positions_progress
        )

        all_positions = batch_result.get("all_positions", [])

        # Enrich positions with trader credentials
        enriched_positions = self._enrich_positions_with_trader_data(
            all_positions, elite_traders
        )

        logger.info(f"✅ Surveillance complete: {len(enriched_positions)} open positions from {batch_result.get('wallets_with_positions', 0)} active traders")

        return self._generate_positions_report(enriched_positions, elite_traders)

    async def _save_open_position(self, position_data: Dict[str, Any]) -> None:
        """Save open position to database."""
        try:
            self.db.supabase.table("wallet_open_positions").upsert(position_data).execute()
        except Exception as e:
            logger.debug(f"Open position save skipped: {e}")

    def _log_positions_progress(self, wallet: str, current: int, total: int) -> None:
        """Log positions fetch progress."""
        if current % 10 == 0 or current == total:
            logger.info(f"📊 Positions: {current}/{total} wallets ({wallet[:12]}...)")

    def _enrich_positions_with_trader_data(
        self,
        positions: List[Dict[str, Any]],
        traders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich positions with trader performance data."""
        trader_lookup = {t["proxy_wallet"]: t for t in traders}

        enriched = []
        for pos in positions:
            wallet = pos.get("proxy_wallet")
            trader_data = trader_lookup.get(wallet, {})

            enriched_pos = {
                **pos,
                "trader_roi": trader_data.get("roi", 0),
                "trader_win_rate": trader_data.get("win_rate", 0),
                "trader_total_volume": trader_data.get("total_volume", 0),
                "trader_n_positions": trader_data.get("n_positions", 0),
                "trader_tier": trader_data.get("tier")
            }
            enriched.append(enriched_pos)

        return enriched

    def _generate_positions_report(
        self,
        positions: List[Dict[str, Any]],
        traders: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate open positions intelligence report."""
        total_value = sum(p.get("current_value", 0) for p in positions)
        total_unrealized_pnl = sum(p.get("cash_pnl", 0) for p in positions)

        # Group by market
        markets_traded = defaultdict(int)
        for pos in positions:
            condition_id = pos.get("condition_id")
            if condition_id:
                markets_traded[condition_id] += 1

        return {
            "total_positions": len(positions),
            "traders_with_positions": len(set(p["proxy_wallet"] for p in positions)),
            "total_traders_analyzed": len(traders),
            "total_position_value": total_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "unique_markets": len(markets_traded),
            "positions": positions,
            "most_popular_markets": sorted(markets_traded.items(), key=lambda x: x[1], reverse=True)[:10],
            "surveillance_time": datetime.now(timezone.utc).isoformat()
        }

    # ========================================================================
    # PHASE 3: SMART MONEY CONSENSUS ANALYSIS
    # ========================================================================

    async def analyze_smart_money_bets(self) -> Dict[str, Any]:
        """
        Aggregate what elite traders are betting on.

        Builds consensus signals by analyzing position overlap.

        Returns:
            Smart money consensus intelligence
        """
        logger.info("🧠 Smart money consensus analysis initiated")

        # Fetch all open positions from database
        result = self.db.supabase.table("wallet_open_positions").select("*").execute()
        all_positions = result.data

        if not all_positions:
            logger.info("📭 No open positions found in database")
            return self._generate_consensus_report({}, {})

        # Get Tier A wallets
        tier_a_wallets = await self.filter_tier_a_wallets()
        tier_a_set = set(tier_a_wallets)

        # Aggregate by market
        market_consensus = defaultdict(lambda: {
            "total_traders": 0,
            "elite_traders": 0,
            "total_value": 0.0,
            "elite_value": 0.0,
            "avg_entry_price": 0.0,
            "positions": [],
            "market_title": None,
            "event_slug": None
        })

        for pos in all_positions:
            condition_id = pos.get("condition_id")
            if not condition_id:
                continue

            wallet = pos.get("proxy_wallet")
            value = pos.get("current_value", 0)
            is_elite = wallet in tier_a_set

            mc = market_consensus[condition_id]
            mc["total_traders"] += 1
            mc["total_value"] += value
            mc["positions"].append(pos)

            if is_elite:
                mc["elite_traders"] += 1
                mc["elite_value"] += value

            if not mc["market_title"]:
                mc["market_title"] = pos.get("title")
                mc["event_slug"] = pos.get("event_slug")

        # Calculate averages and scores
        for condition_id, mc in market_consensus.items():
            if mc["positions"]:
                avg_price = sum(p.get("avg_price", 0) for p in mc["positions"]) / len(mc["positions"])
                mc["avg_entry_price"] = avg_price
                mc["elite_concentration"] = mc["elite_value"] / mc["total_value"] if mc["total_value"] > 0 else 0

            # Remove raw positions to keep report clean
            del mc["positions"]

        # Sort by elite trader count
        sorted_consensus = dict(
            sorted(
                market_consensus.items(),
                key=lambda x: (x[1]["elite_traders"], x[1]["total_value"]),
                reverse=True
            )
        )

        logger.info(f"✅ Consensus built: {len(sorted_consensus)} markets with smart money activity")

        # Identify top opportunities
        top_opportunities = {k: v for k, v in list(sorted_consensus.items())[:20]}

        return self._generate_consensus_report(sorted_consensus, top_opportunities)

    def _generate_consensus_report(
        self,
        all_markets: Dict[str, Any],
        top_markets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate smart money consensus report."""
        return {
            "total_markets_analyzed": len(all_markets),
            "top_opportunities": top_markets,
            "all_market_consensus": all_markets,
            "analysis_time": datetime.now(timezone.utc).isoformat()
        }

    # ========================================================================
    # PHASE 4: COPY-TRADE SUGGESTIONS
    # ========================================================================

    async def get_copy_trade_suggestions(
        self,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60,
        min_elite_traders: int = 2,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable copy-trade suggestions.

        Combines trader credibility with market consensus to identify
        the highest conviction opportunities.

        Args:
            min_roi: Minimum ROI for trader filtering
            min_win_rate: Minimum win rate for trader filtering
            min_elite_traders: Minimum number of elite traders on same position
            top_n: Number of top suggestions to return

        Returns:
            Ranked list of copy-trade suggestions
        """
        logger.info("🎯 Copy-trade suggestion engine initiated")

        # Get open positions from elite traders
        positions_report = await self.get_best_traders_open_positions(
            min_roi=min_roi,
            min_win_rate=min_win_rate,
            max_traders=100
        )

        all_positions = positions_report.get("positions", [])
        if not all_positions:
            logger.info("📭 No open positions from elite traders")
            return []

        # Aggregate by market
        market_aggregates = defaultdict(lambda: {
            "traders": [],
            "total_value": 0.0,
            "avg_roi": 0.0,
            "avg_win_rate": 0.0,
            "avg_entry_price": 0.0,
            "market_title": None,
            "event_slug": None,
            "condition_id": None,
            "outcome": None
        })

        for pos in all_positions:
            condition_id = pos.get("condition_id")
            if not condition_id:
                continue

            ma = market_aggregates[condition_id]
            ma["condition_id"] = condition_id
            ma["traders"].append({
                "wallet": pos.get("proxy_wallet"),
                "roi": pos.get("trader_roi", 0),
                "win_rate": pos.get("trader_win_rate", 0),
                "position_value": pos.get("current_value", 0),
                "entry_price": pos.get("avg_price", 0)
            })
            ma["total_value"] += pos.get("current_value", 0)

            if not ma["market_title"]:
                ma["market_title"] = pos.get("title")
                ma["event_slug"] = pos.get("event_slug")
                ma["outcome"] = pos.get("outcome")

        # Calculate aggregates and filter
        suggestions = []
        for condition_id, ma in market_aggregates.items():
            n_traders = len(ma["traders"])

            if n_traders < min_elite_traders:
                continue

            # Calculate average metrics
            avg_roi = sum(t["roi"] for t in ma["traders"]) / n_traders
            avg_win_rate = sum(t["win_rate"] for t in ma["traders"]) / n_traders
            avg_entry_price = sum(t["entry_price"] for t in ma["traders"]) / n_traders

            # Calculate conviction score
            # Higher score = more traders + higher average ROI + higher win rate
            conviction_score = (
                0.4 * min(1.0, n_traders / 10) +  # Trader count (max at 10)
                0.3 * avg_roi +  # Average ROI
                0.3 * avg_win_rate  # Average win rate
            )

            suggestion = {
                "condition_id": condition_id,
                "market_title": ma["market_title"],
                "event_slug": ma["event_slug"],
                "outcome": ma["outcome"],
                "n_elite_traders": n_traders,
                "total_position_value": ma["total_value"],
                "avg_trader_roi": avg_roi,
                "avg_trader_win_rate": avg_win_rate,
                "avg_entry_price": avg_entry_price,
                "conviction_score": conviction_score,
                "traders": ma["traders"]
            }
            suggestions.append(suggestion)

        # Sort by conviction score
        suggestions.sort(key=lambda x: x["conviction_score"], reverse=True)

        # Limit to top N
        top_suggestions = suggestions[:top_n]

        logger.info(f"✅ Generated {len(top_suggestions)} copy-trade suggestions")

        if top_suggestions:
            top = top_suggestions[0]
            logger.info(f"🏆 Top suggestion: {top['market_title']}")
            logger.info(f"   {top['n_elite_traders']} elite traders | Avg ROI: {top['avg_trader_roi']*100:.1f}% | Conviction: {top['conviction_score']:.2f}")

        return top_suggestions

    # ========================================================================
    # PHASE 5: TAG CREDIBILITY SCORING
    # ========================================================================

    async def compute_trader_tag_credibility(
        self,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute credibility scores for traders in specific tags/categories.

        Args:
            tags: Specific tags to analyze (None = all tags)

        Returns:
            Tag credibility intelligence report
        """
        logger.info("📊 Tag credibility scoring initiated")

        # Fetch all tag stats
        result = self.db.supabase.table("wallet_tag_stats").select("*").execute()
        all_tag_stats = result.data

        if not all_tag_stats:
            logger.info("📭 No tag statistics found")
            return {"tags_analyzed": 0, "credibility_scores": {}}

        # Group by tag
        stats_by_tag = defaultdict(list)
        for stat in all_tag_stats:
            tag = stat.get("tag")
            if tag:
                if tags is None or tag in tags:
                    stats_by_tag[tag].append(stat)

        logger.info(f"🎯 Computing credibility for {len(stats_by_tag)} tags")

        # Compute credibility scores
        tag_credibility = {}
        scores_saved = 0

        for tag, tag_stats in stats_by_tag.items():
            if len(tag_stats) < 3:  # Need at least 3 wallets for meaningful comparison
                continue

            tag_credibility[tag] = {
                "total_wallets": len(tag_stats),
                "top_performers": []
            }

            # Compute credibility for each wallet in this tag
            for wallet_stats in tag_stats:
                proxy_wallet = wallet_stats.get("proxy_wallet")
                cred_score = self.trader_analytics.compute_tag_credibility(
                    proxy_wallet, tag, wallet_stats, tag_stats
                )

                # Save to database
                await self._save_tag_credibility(cred_score)
                scores_saved += 1

                # Track top performers
                if cred_score["credibility_score"] >= 0.7:
                    tag_credibility[tag]["top_performers"].append({
                        "wallet": proxy_wallet,
                        "score": cred_score["credibility_score"],
                        "rank": cred_score.get("tag_rank")
                    })

            # Sort top performers
            tag_credibility[tag]["top_performers"].sort(
                key=lambda x: x["score"], reverse=True
            )

        logger.info(f"✅ Computed {scores_saved} tag credibility scores across {len(tag_credibility)} tags")

        return {
            "tags_analyzed": len(tag_credibility),
            "scores_computed": scores_saved,
            "tag_credibility": tag_credibility,
            "computation_time": datetime.now(timezone.utc).isoformat()
        }

    async def _save_tag_credibility(self, cred_score: Dict[str, Any]) -> None:
        """Save tag credibility score to database."""
        try:
            self.db.supabase.table("trader_tag_credibility").upsert(cred_score).execute()
        except Exception as e:
            logger.debug(f"Tag credibility save skipped: {e}")

    # ========================================================================
    # COMPLETE ANALYSIS ORCHESTRATION
    # ========================================================================

    async def execute_full_trader_analysis(
        self,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60
    ) -> Dict[str, Any]:
        """
        Execute complete trader analysis pipeline.

        Full mission:
        1. Filter best traders
        2. Fetch their open positions
        3. Analyze smart money consensus
        4. Generate copy-trade suggestions
        5. Compute tag credibility

        Args:
            min_roi: Minimum ROI threshold
            min_win_rate: Minimum win rate threshold

        Returns:
            Complete analysis intelligence report
        """
        logger.info("🚀 ELITE FULL TRADER ANALYSIS PROTOCOL INITIATED")

        # Phase 1: Filter Best Traders
        logger.info("📍 PHASE 1: Best Trader Identification")
        elite_traders = await self.filter_best_traders(min_roi=min_roi, min_win_rate=min_win_rate)

        # Phase 2: Open Positions Surveillance
        logger.info("📍 PHASE 2: Open Positions Intelligence")
        positions_intel = await self.get_best_traders_open_positions(
            min_roi=min_roi, min_win_rate=min_win_rate
        )

        # Phase 3: Smart Money Consensus
        logger.info("📍 PHASE 3: Smart Money Consensus Analysis")
        consensus = await self.analyze_smart_money_bets()

        # Phase 4: Copy Trade Suggestions
        logger.info("📍 PHASE 4: Copy-Trade Signal Generation")
        suggestions = await self.get_copy_trade_suggestions(
            min_roi=min_roi, min_win_rate=min_win_rate
        )

        # Phase 5: Tag Credibility
        logger.info("📍 PHASE 5: Tag Credibility Scoring")
        tag_cred = await self.compute_trader_tag_credibility()

        victory_report = {
            "elite_traders_count": len(elite_traders),
            "open_positions_count": positions_intel.get("total_positions", 0),
            "markets_with_consensus": consensus.get("total_markets_analyzed", 0),
            "copy_trade_suggestions": len(suggestions),
            "tags_analyzed": tag_cred.get("tags_analyzed", 0),
            "top_suggestions": suggestions[:5],
            "elite_traders": elite_traders[:10],
            "mission_complete_time": datetime.now(timezone.utc).isoformat()
        }

        logger.info("🎯 MISSION ACCOMPLISHED - Full trader analysis complete!")
        logger.info(f"   Elite traders identified: {len(elite_traders)}")
        logger.info(f"   Open positions tracked: {positions_intel.get('total_positions', 0)}")
        logger.info(f"   Copy-trade signals: {len(suggestions)}")

        return victory_report


# ========================================================================
# STANDALONE FUNCTIONS FOR EASY IMPORT
# ========================================================================

async def filter_best_traders(
    min_roi: float = 0.05,
    min_win_rate: float = 0.60,
    min_volume: float = 10000.0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Filter elite traders by performance metrics.

    Args:
        min_roi: Minimum ROI (0.05 = 5%)
        min_win_rate: Minimum win rate (0.60 = 60%)
        min_volume: Minimum volume in USDC
        limit: Maximum traders to return

    Returns:
        List of elite trader profiles
    """
    analyzer = PolymarketTraderAnalyzer()
    return await analyzer.filter_best_traders(min_roi, min_win_rate, min_volume, limit=limit)


async def get_best_traders_positions(
    min_roi: float = 0.05,
    min_win_rate: float = 0.60,
    max_traders: int = 50
) -> Dict[str, Any]:
    """
    Get current open positions from best traders.

    Args:
        min_roi: Minimum ROI threshold
        min_win_rate: Minimum win rate threshold
        max_traders: Maximum traders to analyze

    Returns:
        Open positions intelligence report
    """
    analyzer = PolymarketTraderAnalyzer()
    return await analyzer.get_best_traders_open_positions(min_roi, min_win_rate, max_traders)


async def get_copy_trade_suggestions(
    min_roi: float = 0.05,
    min_win_rate: float = 0.60,
    min_elite_traders: int = 2,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate actionable copy-trade suggestions.

    Args:
        min_roi: Minimum ROI for trader filtering
        min_win_rate: Minimum win rate for trader filtering
        min_elite_traders: Minimum traders on same position
        top_n: Number of top suggestions

    Returns:
        Ranked list of copy-trade opportunities
    """
    analyzer = PolymarketTraderAnalyzer()
    return await analyzer.get_copy_trade_suggestions(
        min_roi, min_win_rate, min_elite_traders, top_n
    )


async def analyze_smart_money_consensus() -> Dict[str, Any]:
    """
    Analyze what elite traders are betting on.

    Returns:
        Smart money consensus intelligence
    """
    analyzer = PolymarketTraderAnalyzer()
    return await analyzer.analyze_smart_money_bets()


async def execute_full_trader_analysis(
    min_roi: float = 0.05,
    min_win_rate: float = 0.60
) -> Dict[str, Any]:
    """
    Execute complete trader analysis pipeline.

    Args:
        min_roi: Minimum ROI threshold
        min_win_rate: Minimum win rate threshold

    Returns:
        Complete analysis intelligence report
    """
    analyzer = PolymarketTraderAnalyzer()
    return await analyzer.execute_full_trader_analysis(min_roi, min_win_rate)
