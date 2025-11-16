"""
Elite Wallet Discovery & Historical Performance Orchestration Engine.

Assertive execution flow:
1. Discover wallets from closed events intelligence
2. Sync historical performance data (closed positions)
3. Compute wallet statistics and eligibility
4. Deploy wallet scores with tier classification
5. Deliver comprehensive intelligence reports
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketGamma, PolymarketDataAPI
from clients.wallet_tracker import WalletTracker

logger = logging.getLogger(__name__)


class PolymarketWalletCollector:
    """
    Elite wallet intelligence orchestration engine.

    Coordinates:
    1. Wallet discovery from closed event trades
    2. Historical performance sync (closed positions)
    3. Global and tag-specific statistics computation
    4. Composite scoring and tier assignment
    5. Market concentration analysis
    """

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.gamma = PolymarketGamma()
        self.data_api = PolymarketDataAPI()
        self.wallet_tracker = WalletTracker(api=self.data_api, gamma=self.gamma)
        self.max_concurrent = 10

    async def _process_batches(self, items: List, process_func: Callable, batch_size: int = None) -> List:
        """Execute batch processing with assertive parallelism."""
        batch_size = batch_size or self.max_concurrent
        return [
            result
            for i in range(0, len(items), batch_size)
            for result in await asyncio.gather(*[process_func(item) for item in items[i:i + batch_size]])
        ]

    # ========================================================================
    # PHASE 1: WALLET DISCOVERY FROM CLOSED EVENTS
    # ========================================================================

    async def sync_wallets_from_closed_events(
        self,
        limit: int = 100,
        save_trades: bool = True
    ) -> Dict[str, Any]:
        """
        Elite wallet discovery protocol from closed events.

        Assertive execution:
        1. Fetch closed events from intelligence database
        2. Discover all wallets that traded in those events
        3. Save wallet profiles and trade history
        4. Deliver discovery intelligence report

        Args:
            limit: Maximum number of closed events to process
            save_trades: Whether to save individual trades to database

        Returns:
            Discovery intelligence report
        """
        logger.info("🚀 Elite wallet discovery protocol initiated")

        # Fetch closed events from database
        closed_events = await self._fetch_closed_events_from_db(limit)
        if not closed_events:
            logger.info("📭 No closed events found in intelligence database")
            return self._generate_discovery_report(0, 0, 0, 0)

        logger.info(f"🎯 Acquired {len(closed_events)} closed events for wallet reconnaissance")

        # Extract event IDs
        event_ids = [str(event.get("id")) for event in closed_events if event.get("id")]
        if not event_ids:
            logger.warning("⚠️ No valid event IDs extracted from closed events")
            return self._generate_discovery_report(0, 0, 0, 0)

        # Discover wallets from event trades
        discovery_result = await self.wallet_tracker.discover_wallets_from_events(
            event_ids=event_ids,
            save_wallet=self._save_wallet_profile if True else None,
            save_trade=self._save_trade_record if save_trades else None
        )

        wallets_discovered = discovery_result.get("wallets_discovered", 0)
        trades_fetched = discovery_result.get("trades_fetched", 0)

        logger.info(f"✅ Discovery complete: {wallets_discovered} wallets from {trades_fetched} trades")

        # Sync closed events to events_closed table
        events_synced = await self._sync_closed_events_to_db(closed_events)

        return self._generate_discovery_report(
            len(event_ids),
            trades_fetched,
            wallets_discovered,
            events_synced
        )

    async def _fetch_closed_events_from_db(self, limit: int) -> List[Dict[str, Any]]:
        """Fetch closed events from intelligence database."""
        logger.info(f"📡 Fetching closed events from database (limit={limit})")

        # First check events_closed table
        result = self.db.supabase.table("events_closed").select("*").limit(limit).execute()

        if result.data:
            logger.info(f"📦 Retrieved {len(result.data)} closed events from events_closed table")
            return result.data

        # Fallback: Check events table for closed status
        logger.info("📡 Checking events table for closed entries...")
        fallback_result = self.db.supabase.table("events").select("*").eq("status", "closed").limit(limit).execute()

        if fallback_result.data:
            logger.info(f"📦 Retrieved {len(fallback_result.data)} closed events from events table")
            return fallback_result.data

        logger.info("📭 No closed events found - initiating fresh sync from Polymarket")
        return await self._fetch_closed_events_from_api(limit)

    async def _fetch_closed_events_from_api(self, limit: int) -> List[Dict[str, Any]]:
        """Fetch closed events directly from Polymarket API."""
        logger.info(f"🌐 Fetching closed events from Polymarket API (limit={limit})")
        closed_events = await self.gamma.get_all_events(
            active=False,
            page_size=100,
            max_events=limit,
            closed="true"
        )
        logger.info(f"📦 Fetched {len(closed_events)} closed events from API")
        return closed_events

    async def _sync_closed_events_to_db(self, events: List[Dict[str, Any]]) -> int:
        """Sync closed events to events_closed table."""
        logger.info(f"💾 Syncing {len(events)} closed events to database")

        synced_count = 0
        for event in events:
            try:
                normalized = self.wallet_tracker._normalize_closed_event(event)
                self.db.supabase.table("events_closed").upsert(normalized).execute()
                synced_count += 1
            except Exception as e:
                logger.debug(f"Event {event.get('id')} sync skipped: {e}")

        logger.info(f"✅ Synced {synced_count}/{len(events)} closed events")
        return synced_count

    async def _save_wallet_profile(self, wallet_data: Dict[str, Any]) -> None:
        """Save wallet profile to database."""
        try:
            self.db.supabase.table("wallets").upsert(wallet_data).execute()
        except Exception as e:
            logger.debug(f"Wallet save skipped: {e}")

    async def _save_trade_record(self, trade_data: Dict[str, Any]) -> None:
        """Save trade record to database."""
        try:
            self.db.supabase.table("trades").upsert(trade_data).execute()
        except Exception as e:
            logger.debug(f"Trade save skipped: {e}")

    def _generate_discovery_report(
        self,
        events_processed: int,
        trades_fetched: int,
        wallets_discovered: int,
        events_synced: int
    ) -> Dict[str, Any]:
        """Generate wallet discovery intelligence report."""
        return {
            "events_processed": events_processed,
            "trades_fetched": trades_fetched,
            "wallets_discovered": wallets_discovered,
            "events_synced": events_synced,
            "discovery_time": datetime.now(timezone.utc).isoformat()
        }

    # ========================================================================
    # PHASE 2: PERFORMANCE DATA SYNC
    # ========================================================================

    async def sync_wallet_performance_data(
        self,
        wallet_addresses: Optional[List[str]] = None,
        max_wallets: int = 100
    ) -> Dict[str, Any]:
        """
        Sync historical performance data for wallets.

        Assertive execution:
        1. Fetch closed positions for each wallet
        2. Compute global wallet statistics
        3. Compute tag-specific statistics
        4. Save all metrics to database

        Args:
            wallet_addresses: Specific wallets to sync (None = fetch from DB)
            max_wallets: Maximum wallets to process

        Returns:
            Performance sync intelligence report
        """
        logger.info("🚀 Elite performance sync protocol initiated")

        # Get wallet addresses if not provided
        if not wallet_addresses:
            wallet_addresses = await self._fetch_wallet_addresses_from_db(max_wallets)

        if not wallet_addresses:
            logger.info("📭 No wallets found for performance sync")
            return self._generate_performance_report(0, 0, 0.0, 0.0)

        logger.info(f"🎯 Processing performance data for {len(wallet_addresses)} wallets")

        # Fetch closed events for event enrichment
        events = await self._fetch_closed_events_from_db(500)
        events_by_id, events_by_slug = self.wallet_tracker.build_events_lookup(events)

        # Sync closed positions for all wallets
        batch_result = await self.wallet_tracker.sync_wallets_closed_positions_batch(
            proxy_wallets=wallet_addresses,
            save_position=self._save_closed_position,
            events_by_slug=events_by_slug,
            max_concurrency=self.max_concurrent,
            progress_callback=self._log_sync_progress
        )

        # Compute and save wallet statistics
        stats_computed = await self._compute_and_save_all_wallet_stats(
            wallet_addresses,
            events_by_slug,
            events_by_id
        )

        return self._generate_performance_report(
            batch_result.get("wallets_processed", 0),
            batch_result.get("total_positions", 0),
            batch_result.get("total_volume", 0.0),
            batch_result.get("total_pnl", 0.0),
            stats_computed
        )

    async def _fetch_wallet_addresses_from_db(self, limit: int) -> List[str]:
        """Fetch wallet addresses from database."""
        logger.info(f"📡 Fetching wallet addresses (limit={limit})")
        result = self.db.supabase.table("wallets").select("proxy_wallet").limit(limit).execute()
        addresses = [w.get("proxy_wallet") for w in result.data if w.get("proxy_wallet")]
        logger.info(f"📦 Retrieved {len(addresses)} wallet addresses")
        return addresses

    async def _save_closed_position(self, position_data: Dict[str, Any]) -> None:
        """Save closed position to database."""
        try:
            self.db.supabase.table("wallet_closed_positions").upsert(position_data).execute()
        except Exception as e:
            logger.debug(f"Position save skipped: {e}")

    def _log_sync_progress(self, wallet: str, current: int, total: int) -> None:
        """Log wallet sync progress."""
        if current % 10 == 0 or current == total:
            logger.info(f"📊 Progress: {current}/{total} wallets ({wallet[:12]}...)")

    async def _compute_and_save_all_wallet_stats(
        self,
        wallet_addresses: List[str],
        events_by_slug: Dict[str, Dict[str, Any]],
        events_by_id: Dict[str, Dict[str, Any]]
    ) -> int:
        """Compute and save global and tag statistics for all wallets."""
        logger.info(f"📊 Computing statistics for {len(wallet_addresses)} wallets")

        stats_saved = 0

        async def compute_wallet_stats(wallet: str) -> bool:
            # Fetch closed positions from database
            positions = await self._fetch_wallet_positions_from_db(wallet)
            if not positions:
                return False

            # Compute global stats
            global_stats = self.wallet_tracker.compute_wallet_stats_from_positions(positions)
            global_stats["proxy_wallet"] = wallet
            await self._save_wallet_stats(global_stats)

            # Compute tag-specific stats
            tag_stats = self.wallet_tracker.compute_wallet_tag_stats_with_ids(
                wallet, positions, events_by_slug, events_by_id
            )
            for tag_stat in tag_stats:
                await self._save_wallet_tag_stats(tag_stat)

            # Compute market stats
            market_stats = self.wallet_tracker.compute_wallet_market_stats(wallet, positions)
            for market_stat in market_stats:
                await self._save_wallet_market_stats(market_stat)

            return True

        results = await self._process_batches(wallet_addresses, compute_wallet_stats, batch_size=20)
        stats_saved = sum(results)

        logger.info(f"✅ Computed statistics for {stats_saved}/{len(wallet_addresses)} wallets")
        return stats_saved

    async def _fetch_wallet_positions_from_db(self, wallet: str) -> List[Dict[str, Any]]:
        """Fetch closed positions for a wallet from database."""
        result = self.db.supabase.table("wallet_closed_positions").select("*").eq("proxy_wallet", wallet).execute()
        return result.data

    async def _save_wallet_stats(self, stats: Dict[str, Any]) -> None:
        """Save global wallet stats to database."""
        try:
            self.db.supabase.table("wallet_stats").upsert(stats).execute()
        except Exception as e:
            logger.debug(f"Wallet stats save skipped: {e}")

    async def _save_wallet_tag_stats(self, tag_stats: Dict[str, Any]) -> None:
        """Save tag-specific stats to database."""
        try:
            self.db.supabase.table("wallet_tag_stats").upsert(tag_stats).execute()
        except Exception as e:
            logger.debug(f"Tag stats save skipped: {e}")

    async def _save_wallet_market_stats(self, market_stats: Dict[str, Any]) -> None:
        """Save market-specific stats to database."""
        try:
            self.db.supabase.table("wallet_market_stats").upsert(market_stats).execute()
        except Exception as e:
            logger.debug(f"Market stats save skipped: {e}")

    async def update_wallet_aggregates_from_trades(self) -> Dict[str, Any]:
        """
        Update wallet aggregate stats (total_trades, total_volume, total_markets)
        from the trades table.

        Returns:
            Update summary with counts
        """
        logger.info("🚀 Updating wallet aggregates from trades...")

        # Get all trades (paginated)
        all_trades = []
        offset = 0
        batch_size = 1000

        while True:
            result = self.db.supabase.table("trades").select("proxy_wallet,price,size,asset").range(offset, offset + batch_size - 1).execute()
            if not result.data:
                break
            all_trades.extend(result.data)
            if len(result.data) < batch_size:
                break
            offset += batch_size

        logger.info(f"📦 Loaded {len(all_trades)} trades for aggregation")

        # Aggregate by wallet
        wallet_aggregates = {}
        for trade in all_trades:
            wallet = trade.get("proxy_wallet")
            if not wallet:
                continue

            if wallet not in wallet_aggregates:
                wallet_aggregates[wallet] = {
                    "total_trades": 0,
                    "total_volume": 0.0,
                    "markets": set()
                }

            wallet_aggregates[wallet]["total_trades"] += 1
            # Volume = price * size
            price = trade.get("price", 0) or 0
            size = trade.get("size", 0) or 0
            wallet_aggregates[wallet]["total_volume"] += float(price) * float(size)

            # Track unique markets (assets)
            asset = trade.get("asset")
            if asset:
                wallet_aggregates[wallet]["markets"].add(asset)

        logger.info(f"📊 Aggregated stats for {len(wallet_aggregates)} unique wallets")

        # Update wallets table
        updated_count = 0
        for wallet, aggregates in wallet_aggregates.items():
            try:
                update_data = {
                    "total_trades": aggregates["total_trades"],
                    "total_volume": aggregates["total_volume"],
                    "total_markets": len(aggregates["markets"]),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                self.db.supabase.table("wallets").update(update_data).eq("proxy_wallet", wallet).execute()
                updated_count += 1
            except Exception as e:
                logger.debug(f"Wallet aggregate update skipped for {wallet[:20]}: {e}")

        logger.info(f"✅ Updated aggregates for {updated_count}/{len(wallet_aggregates)} wallets")

        return {
            "total_trades_processed": len(all_trades),
            "unique_wallets": len(wallet_aggregates),
            "wallets_updated": updated_count,
            "update_time": datetime.now(timezone.utc).isoformat()
        }

    def _generate_performance_report(
        self,
        wallets_processed: int,
        positions_synced: int,
        total_volume: float,
        total_pnl: float,
        stats_computed: int = 0
    ) -> Dict[str, Any]:
        """Generate performance sync intelligence report."""
        return {
            "wallets_processed": wallets_processed,
            "positions_synced": positions_synced,
            "total_volume": total_volume,
            "total_pnl": total_pnl,
            "stats_computed": stats_computed,
            "sync_time": datetime.now(timezone.utc).isoformat()
        }

    # ========================================================================
    # PHASE 3: WALLET SCORING & TIER ASSIGNMENT
    # ========================================================================

    async def compute_and_save_wallet_scores(self) -> Dict[str, Any]:
        """
        Compute composite scores and tier assignments for all eligible wallets.

        Assertive execution:
        1. Fetch all wallet stats from database
        2. Compute composite scores using multi-factor formula
        3. Assign tiers (A/B/C) based on score thresholds
        4. Save scores to wallet_scores table

        Returns:
            Scoring intelligence report
        """
        logger.info("🚀 Elite wallet scoring protocol initiated")

        # Fetch all wallet stats
        all_stats = await self._fetch_all_wallet_stats()
        if not all_stats:
            logger.info("📭 No wallet statistics found for scoring")
            return self._generate_scoring_report(0, 0, 0, 0)

        logger.info(f"🎯 Scoring {len(all_stats)} wallets for tier assignment")

        # Compute scores for each wallet
        scores_computed = 0
        tier_counts = {"A": 0, "B": 0, "C": 0, "ineligible": 0}

        for wallet_stats in all_stats:
            # Get tag stats for context
            tag_stats = await self._fetch_wallet_best_tag_stats(wallet_stats.get("proxy_wallet"))

            # Compute score
            score = self.wallet_tracker.score_wallet(wallet_stats, tag_stats)
            score["proxy_wallet"] = wallet_stats.get("proxy_wallet")

            # Save score
            await self._save_wallet_score(score)

            # Track tier distribution
            tier = score.get("tier")
            if tier in tier_counts:
                tier_counts[tier] += 1
            else:
                tier_counts["ineligible"] += 1

            scores_computed += 1

        logger.info(f"✅ Scored {scores_computed} wallets | Tier A: {tier_counts['A']} | Tier B: {tier_counts['B']} | Tier C: {tier_counts['C']}")

        return self._generate_scoring_report(
            scores_computed,
            tier_counts["A"],
            tier_counts["B"],
            tier_counts["C"]
        )

    async def _fetch_all_wallet_stats(self) -> List[Dict[str, Any]]:
        """Fetch all wallet statistics from database."""
        result = self.db.supabase.table("wallet_stats").select("*").execute()
        return result.data

    async def _fetch_wallet_best_tag_stats(self, proxy_wallet: str) -> Optional[Dict[str, Any]]:
        """Fetch the best performing tag stats for a wallet."""
        result = self.db.supabase.table("wallet_tag_stats").select("*").eq("proxy_wallet", proxy_wallet).order("roi", desc=True).limit(1).execute()
        return result.data[0] if result.data else None

    async def _save_wallet_score(self, score: Dict[str, Any]) -> None:
        """Save wallet score to database."""
        try:
            self.db.supabase.table("wallet_scores").upsert(score).execute()
        except Exception as e:
            logger.debug(f"Score save skipped: {e}")

    def _generate_scoring_report(
        self,
        scores_computed: int,
        tier_a_count: int,
        tier_b_count: int,
        tier_c_count: int
    ) -> Dict[str, Any]:
        """Generate scoring intelligence report."""
        return {
            "scores_computed": scores_computed,
            "tier_a_count": tier_a_count,
            "tier_b_count": tier_b_count,
            "tier_c_count": tier_c_count,
            "elite_wallets": tier_a_count,
            "scoring_time": datetime.now(timezone.utc).isoformat()
        }

    # ========================================================================
    # COMPLETE ORCHESTRATION
    # ========================================================================

    async def execute_full_wallet_sync(self, event_limit: int = 100) -> Dict[str, Any]:
        """
        Execute complete wallet intelligence acquisition pipeline.

        Full mission:
        1. Discover wallets from closed events
        2. Sync performance data
        3. Compute and save scores

        Args:
            event_limit: Maximum events to process

        Returns:
            Complete mission report
        """
        logger.info("🚀 ELITE FULL WALLET SYNC PROTOCOL INITIATED")

        # Phase 1: Discovery
        logger.info("📍 PHASE 1: Wallet Discovery")
        discovery = await self.sync_wallets_from_closed_events(limit=event_limit)

        # Phase 2: Performance Sync
        logger.info("📍 PHASE 2: Performance Data Sync")
        performance = await self.sync_wallet_performance_data(max_wallets=500)

        # Phase 3: Scoring
        logger.info("📍 PHASE 3: Wallet Scoring & Tier Assignment")
        scoring = await self.compute_and_save_wallet_scores()

        victory_report = {
            "discovery": discovery,
            "performance": performance,
            "scoring": scoring,
            "mission_complete_time": datetime.now(timezone.utc).isoformat()
        }

        logger.info("🎯 MISSION ACCOMPLISHED - Full wallet intelligence pipeline complete!")
        logger.info(f"   Wallets discovered: {discovery.get('wallets_discovered', 0)}")
        logger.info(f"   Positions synced: {performance.get('positions_synced', 0)}")
        logger.info(f"   Elite (Tier A) wallets: {scoring.get('tier_a_count', 0)}")

        return victory_report


# ========================================================================
# STANDALONE FUNCTIONS FOR EASY IMPORT
# ========================================================================

async def sync_wallets_from_closed_events(limit: int = 100) -> Dict[str, Any]:
    """
    Elite wallet discovery from closed events.

    Args:
        limit: Maximum closed events to process

    Returns:
        Discovery intelligence report
    """
    collector = PolymarketWalletCollector()
    return await collector.sync_wallets_from_closed_events(limit=limit)


async def sync_wallet_performance(wallet_addresses: Optional[List[str]] = None, max_wallets: int = 100) -> Dict[str, Any]:
    """
    Sync historical performance data for wallets.

    Args:
        wallet_addresses: Specific wallets (None = fetch from DB)
        max_wallets: Maximum wallets to process

    Returns:
        Performance sync report
    """
    collector = PolymarketWalletCollector()
    return await collector.sync_wallet_performance_data(wallet_addresses, max_wallets)


async def compute_wallet_scores() -> Dict[str, Any]:
    """
    Compute and save wallet scores with tier assignments.

    Returns:
        Scoring intelligence report
    """
    collector = PolymarketWalletCollector()
    return await collector.compute_and_save_wallet_scores()


async def execute_full_wallet_sync(event_limit: int = 100) -> Dict[str, Any]:
    """
    Execute complete wallet intelligence acquisition pipeline.

    Args:
        event_limit: Maximum events to process

    Returns:
        Complete mission report
    """
    collector = PolymarketWalletCollector()
    return await collector.execute_full_wallet_sync(event_limit=event_limit)
