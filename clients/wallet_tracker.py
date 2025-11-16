"""
Polymarket Wallet Tracker & Copy-Trading Analysis Client

Comprehensive wallet tracking system that:
1. Discovers wallets from event trades
2. Syncs wallet performance data (trades, PnL, positions)
3. Computes wallet statistics (ROI, win rate by tag/category)
4. Scores wallets for copy-trading eligibility
5. Analyzes market concentration & smart money flow

API Documentation: https://docs.polymarket.com/developers/misc-endpoints/
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from clients.polymarket import PolymarketDataAPI, PolymarketGamma
from database.client import MarketDatabase

logger = logging.getLogger(__name__)


class WalletTracker:
    """
    High-level wallet tracking and copy-trading analysis orchestrator.

    Coordinates wallet discovery, performance sync, statistics computation,
    market concentration analysis, and copy-trading scoring.
    """

    def __init__(
        self,
        api: Optional[PolymarketDataAPI] = None,
        gamma: Optional[PolymarketGamma] = None,
        min_volume: float = 10000.0,
        min_markets: int = 20,
        min_win_rate: float = 0.40
    ):
        """
        Initialize WalletTracker.

        Args:
            api: PolymarketDataAPI instance (creates default if None)
            gamma: PolymarketGamma instance for event fetching (creates default if None)
            min_volume: Minimum total volume for eligibility ($)
            min_markets: Minimum number of markets traded
            min_win_rate: Minimum win rate (0-1)
        """
        self.api = api or PolymarketDataAPI()
        self.gamma = gamma or PolymarketGamma()
        self.min_volume = min_volume
        self.min_markets = min_markets
        self.min_win_rate = min_win_rate

        # Cache for event_slug → event_id resolution
        self._event_slug_to_id_cache: Dict[str, str] = {}

    # PHASE 1: WALLET DISCOVERY

    async def discover_wallets_from_events(
        self,
        event_ids: List[str],
        save_wallet: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_trade: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_trades_per_event: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Discover wallets from trades in specified events asynchronously.

        Args:
            event_ids: List of event IDs to scan
            save_wallet: Callback to save wallet to DB
            save_trade: Callback to save trade to DB
            max_trades_per_event: Limit trades fetched per event (for testing)

        Returns:
            Discovery summary with counts and wallet set
        """
        assert event_ids, "event_ids cannot be empty"
        logger.info(f"🔍 Discovering wallets from {len(event_ids)} events")

        semaphore = asyncio.Semaphore(10)

        async def process_event(event_id: str) -> Tuple[int, Set[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
            async with semaphore:
                wallets, wallet_data, trade_data = set(), [], []
                offset, event_trades = 0, 0

                while True:
                    # Use maximum API limit for faster fetching
                    trades = await self.api.get_trades(event_id=[event_id], limit=10000, offset=offset)
                    if not trades:
                        break

                    for trade in trades:
                        if proxy_wallet := trade.get("proxyWallet"):
                            wallets.add(proxy_wallet)
                            if save_wallet:
                                wallet_data.append(self._normalize_wallet_from_trade(trade))
                            if save_trade:
                                trade_data.append(self._normalize_trade(trade, event_id))

                    event_trades += len(trades)
                    if max_trades_per_event and event_trades >= max_trades_per_event:
                        break
                    if len(trades) < 10000:
                        break
                    offset += 10000

                return event_trades, wallets, wallet_data, trade_data

        # Process all events concurrently
        results = await asyncio.gather(*[process_event(eid) for eid in event_ids], return_exceptions=True)

        # Aggregate results
        all_wallets, total_trades, all_wallet_data, all_trade_data = set(), 0, [], []
        successful_events = sum(1 for r in results if not isinstance(r, Exception))

        for result in results:
            if isinstance(result, Exception):
                continue
            trades, wallets, w_data, t_data = result
            all_wallets.update(wallets)
            total_trades += trades
            all_wallet_data.extend(w_data)
            all_trade_data.extend(t_data)

        # Batch save data
        if all_wallet_data:
            await self._batch_save(save_wallet, all_wallet_data, 500, "wallets")
        if all_trade_data:
            await self._batch_save(save_trade, all_trade_data, 1000, "trades")

        logger.info(f"✅ Discovery complete: {len(all_wallets)} wallets from {total_trades} trades")
        return {
            "events_scanned": len(event_ids),
            "trades_fetched": total_trades,
            "wallets_discovered": len(all_wallets),
            "wallets": all_wallets
        }

    async def _batch_save(self, save_func: Callable, data: List[Dict[str, Any]], batch_size: int, label: str) -> None:
        """Batch save data asynchronously."""
        if not data or not save_func:
            return

        logger.info(f"💾 Saving {len(data)} {label} records in batches of {batch_size}")
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            await asyncio.gather(*[save_func(item) for item in batch])

    async def filter_eligible_wallets(
        self,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60,
        min_volume: float = 10000.0,
        min_trades: int = 20,
        db_client: Optional[MarketDatabase] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter wallets meeting eligibility criteria for copy trading.

        Args:
            min_roi: Minimum ROI threshold
            min_win_rate: Minimum win rate (0-1)
            min_volume: Minimum total volume
            min_trades: Minimum number of trades
            db_client: Database client (creates default if None)

        Returns:
            List of eligible wallet dictionaries with stats
        """
        logger.info(f"Filtering wallets: ROI>={min_roi}, WinRate>={min_win_rate}, Volume>={min_volume}, Trades>={min_trades}")

        if db_client is None:
            from config.settings import Settings
            settings = Settings()
            db_client = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)

        query = db_client.supabase.table("wallet_stats").select(
            "proxy_wallet, total_volume, realized_pnl, n_positions, n_wins, n_markets, n_events"
        ).gte("total_volume", min_volume).gte("n_positions", min_trades)

        result = query.execute()
        eligible_wallets = []

        for wallet in result.data:
            total_volume = wallet.get("total_volume", 0)
            realized_pnl = wallet.get("realized_pnl", 0)
            n_positions = wallet.get("n_positions", 0)
            n_wins = wallet.get("n_wins", 0)

            roi = realized_pnl / total_volume if total_volume > 0 else 0
            win_rate = n_wins / n_positions if n_positions > 0 else 0

            if roi >= min_roi and win_rate >= min_win_rate:
                eligible_wallets.append({
                    "proxy_wallet": wallet["proxy_wallet"],
                    "total_volume": total_volume,
                    "realized_pnl": realized_pnl,
                    "roi": roi,
                    "win_rate": win_rate,
                    "n_positions": n_positions,
                    "n_wins": n_wins,
                    "n_markets": wallet.get("n_markets", 0),
                    "n_events": wallet.get("n_events", 0)
                })

        logger.info(f"Found {len(eligible_wallets)} eligible wallets")
        return eligible_wallets

    async def get_wallet_current_positions(self, proxy_wallet: str, **filters) -> Dict[str, Any]:
        """
        Get current open positions for a wallet.

        Args:
            proxy_wallet: Wallet address
            **filters: Optional filters

        Returns:
            Positions summary dict
        """
        assert proxy_wallet, "proxy_wallet required"
        logger.info(f"Fetching current positions for {proxy_wallet}")

        positions = []
        offset = 0

        while True:
            batch = await self.api.get_positions(user=proxy_wallet, limit=100, offset=offset, **filters)
            if not batch:
                break

            positions.extend([self._normalize_position(p, proxy_wallet) for p in batch])
            if len(batch) < 100:
                break
            offset += 100

        total_value = sum(p.get("current_value", 0) for p in positions)
        logger.info(f"Found {len(positions)} positions worth ${total_value:,.2f}")

        return {
            "wallet": proxy_wallet,
            "positions": positions,
            "total_value": total_value,
            "total_positions": len(positions)
        }

    async def get_eligible_wallets_with_positions(
        self,
        eligibility_criteria: Optional[Dict[str, Any]] = None,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Get eligible wallets that currently have open positions.

        Args:
            eligibility_criteria: Criteria for wallet eligibility
            **filters: Additional filters

        Returns:
            List of wallets with their current positions
        """
        criteria = eligibility_criteria or {
            "min_roi": 0.05,
            "min_win_rate": 0.60,
            "min_volume": 10000.0,
            "min_trades": 20
        }

        # Get eligible wallets
        eligible_wallets = await self.filter_eligible_wallets(**criteria)

        wallets_with_positions = []

        for wallet_info in eligible_wallets:
            wallet_address = wallet_info.get("proxy_wallet")

            # Check if wallet has current positions
            positions_data = await self.get_wallet_current_positions(wallet_address, **filters)

            if positions_data["total_positions"] > 0:
                # Combine wallet stats with positions
                wallet_with_positions = {
                    **wallet_info,
                    **positions_data
                }
                wallets_with_positions.append(wallet_with_positions)

        logger.info(f"Found {len(wallets_with_positions)} eligible wallets with current positions")

        return wallets_with_positions

    def _normalize_position(self, position: Dict[str, Any], proxy_wallet: str) -> Dict[str, Any]:
        """Normalize position data for consistent format."""
        return {
            "proxy_wallet": proxy_wallet,
            "market_id": position.get("market"),
            "condition_id": position.get("conditionId"),
            "outcome": position.get("outcome"),
            "size": position.get("size", 0),
            "entry_price": position.get("avgPrice", 0),
            "current_price": position.get("price", 0),
            "current_value": position.get("value", 0),
            "pnl": position.get("pnl", 0),
            "event_slug": position.get("eventSlug"),
            "market_slug": position.get("marketSlug"),
            "timestamp": position.get("timestamp"),
            "raw": position
        }

    async def sync_wallet_closed_positions(
        self,
        proxy_wallet: str,
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        event_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Sync closed positions for a wallet.

        Args:
            proxy_wallet: Wallet address
            save_position: Callback to save position to DB
            event_ids: Optional filter for specific events

        Returns:
            Summary dict with position counts and totals
        """
        assert proxy_wallet, "proxy_wallet required"
        logger.info(f"Syncing closed positions for {proxy_wallet}")

        positions = []
        offset = 0

        while True:
            batch = await self.api.get_closed_positions(
                user=proxy_wallet, event_id=event_ids, limit=50, offset=offset
            )
            if not batch:
                break

            for position in batch:
                if save_position:
                    await save_position(self._normalize_closed_position(position, proxy_wallet))
                positions.append(position)

            if len(batch) < 50:
                break
            offset += 50

        total_volume = sum(p.get("totalBought", 0) for p in positions)
        realized_pnl = sum(p.get("realizedPnl", 0) for p in positions)

        logger.info(f"✅ {len(positions)} positions | Volume: ${total_volume:,.2f} | PnL: ${realized_pnl:,.2f}")
        return {
            "wallet": proxy_wallet,
            "positions_fetched": len(positions),
            "total_volume": total_volume,
            "realized_pnl": realized_pnl
        }

    # PHASE 2: STATISTICS COMPUTATION

    def compute_wallet_stats_from_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute global wallet statistics from closed positions.

        Args:
            positions: List of closed position dicts

        Returns:
            Wallet stats dict
        """
        if not positions:
            return {
                "total_volume": 0.0, "realized_pnl": 0.0, "roi": 0.0,
                "n_positions": 0, "n_wins": 0, "n_losses": 0, "win_rate": 0.0,
                "n_markets": 0, "n_events": 0, "is_eligible": False
            }

        # Volume & PnL (handle both API and DB formats)
        total_volume = sum(p.get("totalBought") or p.get("total_bought") or 0 for p in positions)
        realized_pnl = sum(p.get("realizedPnl") or p.get("realized_pnl") or 0 for p in positions)
        roi = (realized_pnl / total_volume) if total_volume > 0 else 0.0

        # Win/loss counts
        n_positions = len(positions)
        n_wins = sum(1 for p in positions if (p.get("realizedPnl") or p.get("realized_pnl") or 0) > 0)
        n_losses = n_positions - n_wins
        win_rate = n_wins / n_positions if n_positions > 0 else 0.0

        # Activity metrics
        n_markets = len(set(p.get("conditionId") or p.get("condition_id") for p in positions if p.get("conditionId") or p.get("condition_id")))
        n_events = len(set(p.get("eventSlug") or p.get("event_slug") for p in positions if p.get("eventSlug") or p.get("event_slug")))

        # Timestamps
        timestamps = [p.get("timestamp") for p in positions if p.get("timestamp")]
        first_trade_at = last_trade_at = None
        if timestamps:
            first_trade_at = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).isoformat()
            last_trade_at = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat()

        # Eligibility
        is_eligible = total_volume >= self.min_volume and n_positions >= self.min_markets and win_rate >= self.min_win_rate

        return {
            "total_volume": total_volume,
            "avg_position_size": total_volume / n_positions if n_positions > 0 else 0.0,
            "realized_pnl": realized_pnl,
            "roi": roi,
            "n_positions": n_positions,
            "n_wins": n_wins,
            "n_losses": n_losses,
            "win_rate": win_rate,
            "n_markets": n_markets,
            "n_events": n_events,
            "first_trade_at": first_trade_at,
            "last_trade_at": last_trade_at,
            "is_eligible": is_eligible,
            "computed_at": datetime.now(timezone.utc).isoformat()
        }

    def compute_wallet_tag_stats(
        self,
        positions: List[Dict[str, Any]],
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None,
        events_by_id: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute wallet statistics grouped by event tags/categories.

        Args:
            positions: List of closed position dicts
            events_by_slug: Map of event_slug → event data
            events_by_id: Map of event_id → event data

        Returns:
            List of tag stats dicts
        """
        if not events_by_slug and not events_by_id:
            return []

        tag_positions: Dict[str, List[Dict[str, Any]]] = {}

        # Group positions by tag
        for position in positions:
            event = None
            event_slug = position.get("eventSlug") or position.get("event_slug")
            if events_by_slug and event_slug:
                event = events_by_slug.get(event_slug)
            elif events_by_id and position.get("event_id"):
                event = events_by_id.get(position.get("event_id"))

            if not event:
                continue

            tags = set(event.get("tags", []))
            if category := event.get("category"):
                tags.add(category)

            for tag in tags:
                tag_positions.setdefault(tag, []).append(position)

        # Compute stats per tag
        tag_stats = []
        for tag, tag_pos in tag_positions.items():
            stats = self.compute_wallet_stats_from_positions(tag_pos)
            stats["tag"] = tag
            tag_stats.append(stats)

        logger.info(f"Computed stats for {len(tag_stats)} tags")
        return tag_stats

    # PHASE 3: MARKET CONCENTRATION ANALYSIS

    def compute_market_wallet_distribution(self, trades: List[Dict[str, Any]], condition_id: str) -> Dict[str, Any]:
        """Compute wallet volume distribution for a market."""
        if not trades:
            return {"condition_id": condition_id, "n_wallets": 0, "total_volume": 0.0, "wallet_shares": {}}

        wallet_volumes = {}
        for trade in trades:
            if wallet := trade.get("proxyWallet"):
                notional = trade.get("size", 0) * trade.get("price", 0)
                wallet_volumes[wallet] = wallet_volumes.get(wallet, 0) + notional

        total_volume = sum(wallet_volumes.values())
        wallet_shares = {w: v/total_volume if total_volume > 0 else 0 for w, v in wallet_volumes.items()}
        sorted_wallets = sorted(wallet_shares.items(), key=lambda x: x[1], reverse=True)

        herfindahl = sum(share ** 2 for _, share in sorted_wallets)
        top_shares = [sum(share for _, share in sorted_wallets[:i+1]) for i in [0, 4, 9]]

        return {
            "condition_id": condition_id,
            "n_wallets": len(wallet_volumes),
            "total_volume": total_volume,
            "herfindahl_index": herfindahl,
            "top_1_share": top_shares[0] if len(sorted_wallets) > 0 else 0,
            "top_5_share": top_shares[1] if len(sorted_wallets) > 4 else sum(share for _, share in sorted_wallets),
            "top_10_share": top_shares[2] if len(sorted_wallets) > 9 else sum(share for _, share in sorted_wallets),
            "wallet_shares": dict(sorted_wallets),
            "top_wallets": [wallet for wallet, _ in sorted_wallets[:10]]
        }

    def compute_smart_money_metrics(self, market_distribution: Dict[str, Any], wallet_tiers: Dict[str, str]) -> Dict[str, Any]:
        """Compute smart money concentration for a market."""
        wallet_shares = market_distribution.get("wallet_shares", {})
        total_volume = market_distribution.get("total_volume", 0)

        smart_volume = dumb_volume = 0.0
        for wallet, share in wallet_shares.items():
            volume = share * total_volume
            if wallet_tiers.get(wallet) == "A":
                smart_volume += volume
            elif wallet_tiers.get(wallet) == "C":
                dumb_volume += volume

        total_volume = total_volume or 1  # Avoid division by zero
        return {
            "smart_volume": smart_volume,
            "smart_volume_share": smart_volume / total_volume,
            "smart_wallet_count": sum(1 for w in wallet_shares if wallet_tiers.get(w) == "A"),
            "dumb_volume": dumb_volume,
            "dumb_volume_share": dumb_volume / total_volume
        }

    # PHASE 4: WALLET SCORING

    def score_wallet(
        self,
        wallet_stats: Dict[str, Any],
        tag_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute copy-trading score for a wallet.

        Args:
            wallet_stats: Global wallet stats
            tag_stats: Tag-specific stats (optional)

        Returns:
            Score dict with tier assignment
        """
        roi = wallet_stats.get("roi", 0)
        win_rate = wallet_stats.get("win_rate", 0)
        total_volume = wallet_stats.get("total_volume", 0)
        last_trade_at = wallet_stats.get("last_trade_at")

        # Component scores
        roi_score = max(0, min(1, (roi + 1) / 2))  # Map -100% to +100% → 0 to 1
        win_rate_score = win_rate
        volume_score = min(1, total_volume / 100000.0)  # $100k = max score

        # Recency score (decay over 90 days)
        recency_score = 0.5
        if last_trade_at:
            try:
                last_trade = datetime.fromisoformat(last_trade_at.replace("Z", "+00:00"))
                days_ago = (datetime.now(timezone.utc) - last_trade).days
                recency_score = max(0, 1 - (days_ago / 90))
            except:
                pass

        # Tag-specific ROI score
        roi_tag_score = max(0, min(1, (tag_stats.get("roi", 0) + 1) / 2)) if tag_stats else 0.0

        # Composite score
        composite_score = (
            0.4 * roi_score +
            0.3 * win_rate_score +
            0.2 * (roi_tag_score or roi_score) +
            0.1 * recency_score
        )

        # Tier assignment
        meets_thresholds = wallet_stats.get("is_eligible", False)
        tier = None
        if meets_thresholds:
            if composite_score >= 0.7:
                tier = "A"
            elif composite_score >= 0.5:
                tier = "B"
            else:
                tier = "C"

        return {
            "roi_score": roi_score,
            "win_rate_score": win_rate_score,
            "volume_score": volume_score,
            "recency_score": recency_score,
            "roi_tag_score": roi_tag_score,
            "composite_score": composite_score,
            "meets_thresholds": meets_thresholds,
            "tier": tier,
            "computed_at": datetime.now(timezone.utc).isoformat()
        }

    # DATA ENRICHMENT HELPERS
    def enrich_event_ids(self, records: List[Dict[str, Any]], events_by_slug: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich records with event_id by mapping eventSlug → event.id."""
        for record in records:
            event_slug = record.get("eventSlug") or record.get("event_slug")
            if event_slug and not record.get("event_id"):
                if event := events_by_slug.get(event_slug):
                    record["event_id"] = event.get("id")
        return records

    def build_events_lookup(self, events: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Build lookup dicts for events by ID and slug."""
        events_by_id, events_by_slug = {}, {}
        for event in events:
            if event_id := event.get("id"):
                events_by_id[event_id] = event
            if event_slug := event.get("slug"):
                events_by_slug[event_slug] = event
        return events_by_id, events_by_slug

    def populate_event_slug_cache(self, events_by_slug: Dict[str, Dict[str, Any]]) -> None:
        """Populate the internal event_slug → event_id cache."""
        for slug, event in events_by_slug.items():
            if event_id := event.get("id"):
                self._event_slug_to_id_cache[slug] = str(event_id)
        logger.info(f"Populated event slug cache with {len(self._event_slug_to_id_cache)} entries")

    async def resolve_event_id_from_slug(self, event_slug: str) -> Optional[str]:
        """
        Resolve event_id from event_slug, using cache or fetching from API.

        Args:
            event_slug: Event slug to resolve

        Returns:
            Event ID or None if not found
        """
        if not event_slug:
            return None

        # Check cache first
        if event_slug in self._event_slug_to_id_cache:
            return self._event_slug_to_id_cache[event_slug]

        # Fetch from Gamma API
        try:
            event = await self.gamma.get_event(event_slug)
            if event_id := event.get("id"):
                self._event_slug_to_id_cache[event_slug] = str(event_id)
                return str(event_id)
        except Exception as e:
            logger.debug(f"Could not resolve event_id for slug {event_slug}: {e}")

        return None

    async def enrich_event_ids_async(
        self,
        records: List[Dict[str, Any]],
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Enrich records with event_id, using cache and API fallback.

        Args:
            records: List of records with event_slug
            events_by_slug: Optional pre-built lookup dict

        Returns:
            Enriched records with event_id populated where possible
        """
        if events_by_slug:
            self.populate_event_slug_cache(events_by_slug)

        enriched_count = 0
        for record in records:
            event_slug = record.get("eventSlug") or record.get("event_slug")
            if event_slug and not record.get("event_id"):
                event_id = await self.resolve_event_id_from_slug(event_slug)
                if event_id:
                    record["event_id"] = event_id
                    enriched_count += 1

        logger.info(f"Enriched {enriched_count}/{len(records)} records with event_id")
        return records

    # PHASE 5: CLOSED EVENTS SYNC

    async def sync_closed_events(
        self,
        save_event: Callable[[Dict[str, Any]], None],
        limit: int = 500,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Fetch and save closed/resolved events from Polymarket.

        Args:
            save_event: Callback to save event to DB
            limit: Maximum number of events to fetch
            page_size: API page size for pagination

        Returns:
            Summary dict with counts
        """
        logger.info(f"🔍 Syncing closed events (limit={limit})")

        # Fetch closed events from Gamma API
        closed_events = await self.gamma.get_all_events(
            active=False,
            page_size=page_size,
            max_events=limit,
            closed="true"
        )

        logger.info(f"Fetched {len(closed_events)} closed events from Gamma API")

        # Process and save each event
        events_saved = 0
        events_failed = 0
        for event in closed_events:
            normalized = self._normalize_closed_event(event)
            try:
                await save_event(normalized)
                events_saved += 1

                # Update cache
                if slug := normalized.get("slug"):
                    self._event_slug_to_id_cache[slug] = normalized["id"]
            except Exception as e:
                events_failed += 1
                error_msg = str(e).lower()
                if 'row-level security' in error_msg or '42501' in error_msg:
                    logger.error(f"RLS policy violation for event {event.get('id')}: {e}")
                    logger.error("events_closed table has RLS enabled. Disable RLS in Supabase dashboard or recreate table without RLS.")
                else:
                    logger.warning(f"Failed to save event {event.get('id')}: {e}")

        if events_failed > 0:
            logger.warning(f"⚠️  {events_failed} events failed to save (likely RLS policy issue)")
        else:
            logger.info(f"✅ Saved {events_saved} closed events")

        return {
            "events_fetched": len(closed_events),
            "events_saved": events_saved,
            "events_failed": events_failed,
            "synced_at": datetime.now(timezone.utc).isoformat()
        }

    def _normalize_closed_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize closed event data for database storage.

        Args:
            event: Raw event data from Gamma API

        Returns:
            Normalized event dict ready for DB insertion
        """
        # Extract tags
        tags = event.get("tags", [])
        if tags and isinstance(tags[0], dict):
            tag_labels = [tag.get("label") or tag.get("slug", "") for tag in tags if isinstance(tag, dict)]
        else:
            tag_labels = [str(tag) for tag in tags if tag]

        # Extract category
        category = event.get("category")
        if not category and tag_labels:
            category = tag_labels[0]

        # Calculate market metrics
        markets = event.get("markets", [])
        market_count = len(markets)
        total_liquidity = 0.0
        total_volume = 0.0

        for market in markets:
            total_liquidity += float(market.get("liquidityNum") or market.get("liquidity") or 0)
            total_volume += float(market.get("volumeNum") or market.get("volume") or 0)

        return {
            "id": str(event.get("id")),
            "platform": "polymarket",
            "title": event.get("title", ""),
            "description": event.get("description", "")[:1000] if event.get("description") else "",
            "category": category,
            "tags": tag_labels,
            "status": "closed",
            "start_date": event.get("startDate") or event.get("startTime"),
            "end_date": event.get("endDate"),
            "market_count": market_count,
            "total_liquidity": total_liquidity,
            "total_volume": total_volume,
            "resolution_source": event.get("resolutionSource"),
            "resolution_date": event.get("resolvedAt"),
            "slug": event.get("slug") or event.get("ticker"),
            "created_at": event.get("createdAt"),
            "updated_at": event.get("updatedAt"),
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": event
        }

    # PHASE 6: CACHE-AWARE SYNCING

    def is_wallet_stale(
        self,
        last_sync_at: Optional[str],
        max_age_hours: int = 24
    ) -> bool:
        """
        Check if wallet data needs refresh based on last sync time.

        Args:
            last_sync_at: ISO timestamp of last sync
            max_age_hours: Maximum age in hours before data is stale

        Returns:
            True if data is stale and needs refresh
        """
        if not last_sync_at:
            return True

        try:
            last_sync = datetime.fromisoformat(last_sync_at.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - last_sync).total_seconds() / 3600
            return age_hours >= max_age_hours
        except Exception:
            return True

    async def sync_wallet_closed_positions_with_enrichment(
        self,
        proxy_wallet: str,
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None,
        event_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Sync closed positions with automatic event_id enrichment.

        Args:
            proxy_wallet: Wallet address
            save_position: Callback to save position to DB
            events_by_slug: Optional event lookup for enrichment
            event_ids: Optional filter for specific events

        Returns:
            Summary dict with position counts and totals
        """
        assert proxy_wallet, "proxy_wallet required"
        logger.info(f"Syncing closed positions (with enrichment) for {proxy_wallet}")

        # Populate cache if events provided
        if events_by_slug:
            self.populate_event_slug_cache(events_by_slug)

        positions = []
        offset = 0

        while True:
            batch = await self.api.get_closed_positions(
                user=proxy_wallet, event_id=event_ids, limit=50, offset=offset
            )
            if not batch:
                break

            for position in batch:
                normalized = self._normalize_closed_position(position, proxy_wallet)

                # Enrich with event_id
                if event_slug := normalized.get("event_slug"):
                    event_id = await self.resolve_event_id_from_slug(event_slug)
                    if event_id:
                        normalized["event_id"] = event_id

                if save_position:
                    await save_position(normalized)
                positions.append(normalized)

            if len(batch) < 50:
                break
            offset += 50

        total_volume = sum(p.get("total_bought", 0) for p in positions)
        realized_pnl = sum(p.get("realized_pnl", 0) for p in positions)

        logger.info(f"✅ {len(positions)} positions (enriched) | Volume: ${total_volume:,.2f} | PnL: ${realized_pnl:,.2f}")
        return {
            "wallet": proxy_wallet,
            "positions_fetched": len(positions),
            "positions_enriched": sum(1 for p in positions if p.get("event_id")),
            "total_volume": total_volume,
            "realized_pnl": realized_pnl
        }

    async def sync_wallet_open_positions(
        self,
        proxy_wallet: str,
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Sync current open positions for a wallet.

        Args:
            proxy_wallet: Wallet address
            save_position: Callback to save position to DB
            events_by_slug: Optional event lookup for enrichment

        Returns:
            Summary dict with position counts
        """
        assert proxy_wallet, "proxy_wallet required"
        logger.info(f"Syncing open positions for {proxy_wallet}")

        # Populate cache if events provided
        if events_by_slug:
            self.populate_event_slug_cache(events_by_slug)

        positions = []
        offset = 0

        while True:
            batch = await self.api.get_positions(user=proxy_wallet, limit=100, offset=offset)
            if not batch:
                break

            for position in batch:
                normalized = self._normalize_open_position(position, proxy_wallet)

                # Enrich with event_id
                if event_slug := normalized.get("event_slug"):
                    event_id = await self.resolve_event_id_from_slug(event_slug)
                    if event_id:
                        normalized["event_id"] = event_id

                if save_position:
                    await save_position(normalized)
                positions.append(normalized)

            if len(batch) < 100:
                break
            offset += 100

        total_value = sum(p.get("current_value", 0) for p in positions)
        unrealized_pnl = sum(p.get("cash_pnl", 0) for p in positions)

        logger.info(f"✅ {len(positions)} open positions | Value: ${total_value:,.2f} | Unrealized PnL: ${unrealized_pnl:,.2f}")
        return {
            "wallet": proxy_wallet,
            "positions_fetched": len(positions),
            "positions_enriched": sum(1 for p in positions if p.get("event_id")),
            "total_value": total_value,
            "unrealized_pnl": unrealized_pnl
        }

    # ========================================================================
    # NORMALIZATION HELPERS
    # ========================================================================

    def _normalize_wallet_from_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Extract wallet profile from trade data."""
        timestamp = trade.get("timestamp", 0)
        timestamp_iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat() if timestamp else None
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "proxy_wallet": trade.get("proxyWallet"),
            "name": trade.get("name") or "",
            "pseudonym": trade.get("pseudonym") or "",
            "bio": trade.get("bio") or "",
            "profile_image": trade.get("profileImage") or "",
            "profile_image_optimized": trade.get("profileImageOptimized") or "",
            "display_username_public": trade.get("displayUsernamePublic", False),
            "first_seen_at": timestamp_iso,
            "last_seen_at": timestamp_iso,
            "last_sync_at": now_iso,
            "total_trades": 0,
            "total_markets": 0,
            "total_volume": 0.0,
            "created_at": now_iso,
            "updated_at": now_iso,
            "raw_data": trade
        }

    def _normalize_trade(self, trade: Dict[str, Any], event_id: Optional[str] = None) -> Dict[str, Any]:
        """Normalize trade data."""
        tx_hash = trade.get("transactionHash", "")
        asset = trade.get("asset", "")
        timestamp = trade.get("timestamp", 0)
        trade_id = hashlib.sha256(f"{tx_hash}{asset}{timestamp}".encode()).hexdigest()[:32]
        size, price = trade.get("size", 0), trade.get("price", 0)
        return {
            "id": trade_id,
            "proxy_wallet": trade.get("proxyWallet"),
            "event_id": event_id,
            "condition_id": trade.get("conditionId"),
            "side": trade.get("side"),
            "asset": asset,
            "size": size,
            "price": price,
            "notional": size * price,
            "timestamp": timestamp,
            "transaction_hash": tx_hash,
            "title": trade.get("title"),
            "slug": trade.get("slug"),
            "event_slug": trade.get("eventSlug"),
            "outcome": trade.get("outcome"),
            "outcome_index": trade.get("outcomeIndex"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": trade
        }

    def _normalize_closed_position(self, position: Dict[str, Any], proxy_wallet: str) -> Dict[str, Any]:
        """Normalize closed position data."""
        condition_id = position.get("conditionId", "")
        outcome_index = position.get("outcomeIndex", 0)
        pos_id = hashlib.sha256(f"{proxy_wallet}{condition_id}{outcome_index}".encode()).hexdigest()[:32]
        return {
            "id": pos_id,
            "proxy_wallet": proxy_wallet,
            "event_id": None,
            "condition_id": condition_id,
            "asset": position.get("asset"),
            "outcome": position.get("outcome"),
            "outcome_index": outcome_index,
            "total_bought": position.get("totalBought", 0),
            "avg_price": position.get("avgPrice", 0),
            "cur_price": position.get("curPrice", 0),
            "realized_pnl": position.get("realizedPnl", 0),
            "timestamp": position.get("timestamp", 0),
            "end_date": position.get("endDate"),
            "title": position.get("title"),
            "slug": position.get("slug"),
            "event_slug": position.get("eventSlug"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": position
        }

    def _normalize_position(self, position: Dict[str, Any], proxy_wallet: str) -> Dict[str, Any]:
        """Normalize open position data."""
        condition_id = position.get("conditionId", "")
        outcome_index = position.get("outcomeIndex", 0)
        pos_id = hashlib.sha256(f"{proxy_wallet}{condition_id}{outcome_index}".encode()).hexdigest()[:32]
        return {
            "id": pos_id,
            "proxy_wallet": proxy_wallet,
            "condition_id": condition_id,
            "outcome": position.get("outcome"),
            "size": position.get("size", 0),
            "entry_price": position.get("avgPrice", 0),
            "current_price": position.get("price", 0),
            "current_value": position.get("value", 0),
            "pnl": position.get("pnl", 0),
            "event_slug": position.get("eventSlug"),
            "market_slug": position.get("marketSlug"),
            "timestamp": position.get("timestamp"),
            "raw": position
        }

    def _normalize_open_position(self, position: Dict[str, Any], proxy_wallet: str) -> Dict[str, Any]:
        """
        Normalize open position data for database storage.

        Captures all fields from /positions API endpoint including P&L metrics
        and position flags (redeemable, mergeable, negativeRisk).

        Args:
            position: Raw position data from Polymarket API
            proxy_wallet: Wallet address

        Returns:
            Normalized position dict ready for DB insertion
        """
        condition_id = position.get("conditionId", "")
        outcome_index = position.get("outcomeIndex", 0)
        pos_id = hashlib.sha256(f"{proxy_wallet}{condition_id}{outcome_index}".encode()).hexdigest()[:32]

        size = float(position.get("size", 0))
        avg_price = float(position.get("avgPrice", 0))
        cur_price = float(position.get("curPrice", 0))

        initial_value = size * avg_price
        current_value = size * cur_price
        cash_pnl = current_value - initial_value
        percent_pnl = (cash_pnl / initial_value * 100) if initial_value > 0 else 0.0

        return {
            "id": pos_id,
            "proxy_wallet": proxy_wallet,
            "event_id": None,  # Will be enriched later
            "condition_id": condition_id,
            "asset": position.get("asset"),
            "outcome": position.get("outcome"),
            "outcome_index": outcome_index,
            "size": size,
            "avg_price": avg_price,
            "cur_price": cur_price,
            "initial_value": initial_value,
            "current_value": current_value,
            "cash_pnl": cash_pnl,
            "percent_pnl": percent_pnl,
            "realized_pnl": float(position.get("realizedPnl", 0)),
            "percent_realized_pnl": float(position.get("percentRealizedPnl", 0)),
            "redeemable": bool(position.get("redeemable", False)),
            "mergeable": bool(position.get("mergeable", False)),
            "negative_risk": bool(position.get("negativeRisk", False)),
            "title": position.get("title"),
            "slug": position.get("slug"),
            "event_slug": position.get("eventSlug"),
            "end_date": position.get("endDate"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "raw_data": position
        }

    # ========================================================================
    # PHASE 7: HIGH-PERFORMANCE BATCH OPERATIONS
    # ========================================================================

    async def sync_wallets_closed_positions_batch(
        self,
        proxy_wallets: List[str],
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None,
        max_concurrency: int = 5,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Sync closed positions for multiple wallets concurrently.

        High-performance batch processing with configurable concurrency control.
        Uses asyncio.Semaphore to limit parallel API requests while maximizing throughput.

        Args:
            proxy_wallets: List of wallet addresses to sync
            save_position: Callback to save position to DB
            events_by_slug: Optional event lookup for enrichment (pre-populates cache)
            max_concurrency: Maximum parallel wallet syncs (default: 5)
            progress_callback: Optional callback(wallet, current_idx, total) for progress updates

        Returns:
            Aggregated summary dict with totals and per-wallet results
        """
        if not proxy_wallets:
            return {
                "wallets_processed": 0,
                "wallets_with_positions": 0,
                "total_positions": 0,
                "total_enriched": 0,
                "total_volume": 0.0,
                "total_pnl": 0.0,
                "results": {}
            }

        logger.info(f"🚀 Batch syncing closed positions for {len(proxy_wallets)} wallets (concurrency={max_concurrency})")

        # Pre-populate event slug cache if provided
        if events_by_slug:
            self.populate_event_slug_cache(events_by_slug)

        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        total_wallets = len(proxy_wallets)

        async def process_wallet(wallet: str, idx: int) -> Tuple[str, Dict[str, Any]]:
            """Process single wallet with semaphore control."""
            async with semaphore:
                if progress_callback:
                    progress_callback(wallet, idx, total_wallets)

                try:
                    result = await self.sync_wallet_closed_positions_with_enrichment(
                        proxy_wallet=wallet,
                        save_position=save_position,
                        events_by_slug=None  # Already cached
                    )
                    return wallet, result
                except Exception as e:
                    logger.warning(f"Error syncing wallet {wallet[:12]}...: {e}")
                    return wallet, {
                        "wallet": wallet,
                        "positions_fetched": 0,
                        "positions_enriched": 0,
                        "total_volume": 0.0,
                        "realized_pnl": 0.0,
                        "error": str(e)
                    }

        # Process all wallets concurrently with controlled parallelism
        tasks = [process_wallet(wallet, idx) for idx, wallet in enumerate(proxy_wallets, 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        aggregated = {
            "wallets_processed": 0,
            "wallets_with_positions": 0,
            "total_positions": 0,
            "total_enriched": 0,
            "total_volume": 0.0,
            "total_pnl": 0.0,
            "results": {}
        }

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected error in batch processing: {result}")
                continue

            wallet, wallet_result = result
            aggregated["wallets_processed"] += 1
            aggregated["results"][wallet] = wallet_result

            if wallet_result.get("positions_fetched", 0) > 0:
                aggregated["wallets_with_positions"] += 1
                aggregated["total_positions"] += wallet_result["positions_fetched"]
                aggregated["total_enriched"] += wallet_result.get("positions_enriched", 0)
                aggregated["total_volume"] += wallet_result.get("total_volume", 0.0)
                aggregated["total_pnl"] += wallet_result.get("realized_pnl", 0.0)

        enrichment_rate = (
            aggregated["total_enriched"] / aggregated["total_positions"] * 100
            if aggregated["total_positions"] > 0 else 0
        )

        logger.info(
            f"✅ Batch sync complete: {aggregated['wallets_with_positions']}/{aggregated['wallets_processed']} "
            f"wallets with positions | {aggregated['total_positions']:,} total positions | "
            f"{enrichment_rate:.1f}% enrichment rate"
        )

        return aggregated

    async def sync_wallets_open_positions_batch(
        self,
        proxy_wallets: List[str],
        save_position: Optional[Callable[[Dict[str, Any]], None]] = None,
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None,
        max_concurrency: int = 5,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Sync current open positions for multiple wallets concurrently.

        Fetches what the best wallets are currently betting on.

        Args:
            proxy_wallets: List of wallet addresses to sync
            save_position: Callback to save position to DB
            events_by_slug: Optional event lookup for enrichment
            max_concurrency: Maximum parallel wallet syncs (default: 5)
            progress_callback: Optional callback(wallet, current_idx, total) for progress

        Returns:
            Aggregated summary with all open positions and totals
        """
        if not proxy_wallets:
            return {
                "wallets_processed": 0,
                "wallets_with_positions": 0,
                "total_positions": 0,
                "total_value": 0.0,
                "total_unrealized_pnl": 0.0,
                "all_positions": [],
                "results": {}
            }

        logger.info(f"🚀 Batch syncing open positions for {len(proxy_wallets)} wallets (concurrency={max_concurrency})")

        # Pre-populate event slug cache
        if events_by_slug:
            self.populate_event_slug_cache(events_by_slug)

        semaphore = asyncio.Semaphore(max_concurrency)
        total_wallets = len(proxy_wallets)

        async def process_wallet(wallet: str, idx: int) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
            """Process single wallet for open positions."""
            async with semaphore:
                if progress_callback:
                    progress_callback(wallet, idx, total_wallets)

                try:
                    result = await self.sync_wallet_open_positions(
                        proxy_wallet=wallet,
                        save_position=save_position,
                        events_by_slug=None  # Already cached
                    )

                    # Fetch the actual positions for aggregation
                    positions = []
                    offset = 0
                    while True:
                        batch = await self.api.get_positions(user=wallet, limit=100, offset=offset)
                        if not batch:
                            break
                        for pos in batch:
                            normalized = self._normalize_open_position(pos, wallet)
                            # Enrich with event_id
                            if event_slug := normalized.get("event_slug"):
                                event_id = await self.resolve_event_id_from_slug(event_slug)
                                if event_id:
                                    normalized["event_id"] = event_id
                            positions.append(normalized)
                        if len(batch) < 100:
                            break
                        offset += 100

                    return wallet, result, positions
                except Exception as e:
                    logger.warning(f"Error syncing open positions for {wallet[:12]}...: {e}")
                    return wallet, {
                        "wallet": wallet,
                        "positions_fetched": 0,
                        "total_value": 0.0,
                        "unrealized_pnl": 0.0,
                        "error": str(e)
                    }, []

        # Process all wallets concurrently
        tasks = [process_wallet(wallet, idx) for idx, wallet in enumerate(proxy_wallets, 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        aggregated = {
            "wallets_processed": 0,
            "wallets_with_positions": 0,
            "total_positions": 0,
            "total_value": 0.0,
            "total_unrealized_pnl": 0.0,
            "all_positions": [],
            "results": {}
        }

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected error in open positions batch: {result}")
                continue

            wallet, wallet_result, positions = result
            aggregated["wallets_processed"] += 1
            aggregated["results"][wallet] = wallet_result
            aggregated["all_positions"].extend(positions)

            if wallet_result.get("positions_fetched", 0) > 0:
                aggregated["wallets_with_positions"] += 1
                aggregated["total_positions"] += wallet_result["positions_fetched"]
                aggregated["total_value"] += wallet_result.get("total_value", 0.0)
                aggregated["total_unrealized_pnl"] += wallet_result.get("unrealized_pnl", 0.0)

        logger.info(
            f"✅ Open positions batch sync complete: {aggregated['wallets_with_positions']}/{aggregated['wallets_processed']} "
            f"wallets with positions | {aggregated['total_positions']:,} total positions | "
            f"Value: ${aggregated['total_value']:,.2f}"
        )

        return aggregated

    def compute_wallet_tag_stats_with_ids(
        self,
        proxy_wallet: str,
        positions: List[Dict[str, Any]],
        events_by_slug: Optional[Dict[str, Dict[str, Any]]] = None,
        events_by_id: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute wallet statistics grouped by event tags with proper IDs for persistence.

        Args:
            proxy_wallet: Wallet address
            positions: List of closed position dicts
            events_by_slug: Map of event_slug → event data
            events_by_id: Map of event_id → event data

        Returns:
            List of tag stats dicts ready for database persistence
        """
        if not events_by_slug and not events_by_id:
            return []

        tag_positions: Dict[str, List[Dict[str, Any]]] = {}

        # Group positions by tag
        for position in positions:
            event = None
            event_slug = position.get("eventSlug") or position.get("event_slug")
            if events_by_slug and event_slug:
                event = events_by_slug.get(event_slug)
            elif events_by_id and position.get("event_id"):
                event = events_by_id.get(position.get("event_id"))

            if not event:
                continue

            tags = set(event.get("tags", []))
            if category := event.get("category"):
                tags.add(category)

            for tag in tags:
                tag_positions.setdefault(tag, []).append(position)

        # Compute stats per tag with proper IDs
        tag_stats = []
        for tag, tag_pos in tag_positions.items():
            stats = self.compute_wallet_stats_from_positions(tag_pos)
            # Generate unique ID for this wallet+tag combo
            tag_id = hashlib.sha256(f"{proxy_wallet}_{tag}".encode()).hexdigest()[:32]
            stats["id"] = tag_id
            stats["proxy_wallet"] = proxy_wallet
            stats["tag"] = tag
            # Remove fields that don't belong in tag stats
            stats.pop("is_eligible", None)
            stats.pop("tier", None)
            tag_stats.append(stats)

        logger.debug(f"Computed tag stats for {proxy_wallet[:12]}...: {len(tag_stats)} tags")
        return tag_stats

    def compute_wallet_market_stats(
        self,
        proxy_wallet: str,
        positions: List[Dict[str, Any]],
        trades: Optional[List[Dict[str, Any]]] = None  # Reserved for future use
    ) -> List[Dict[str, Any]]:
        """
        Compute wallet statistics per market (condition_id).

        Analyzes market participation including volume, trade count, and timing.

        Args:
            proxy_wallet: Wallet address
            positions: List of closed positions
            trades: Optional list of trades for more detailed analysis

        Returns:
            List of market stats dicts ready for database persistence
        """
        # Group positions by condition_id
        market_positions: Dict[str, List[Dict[str, Any]]] = {}
        for pos in positions:
            condition_id = pos.get("condition_id") or pos.get("conditionId")
            if condition_id:
                market_positions.setdefault(condition_id, []).append(pos)

        market_stats = []
        for condition_id, market_pos in market_positions.items():
            # Calculate volume and PnL
            wallet_volume = sum(
                p.get("total_bought") or p.get("totalBought") or 0
                for p in market_pos
            )
            realized_pnl = sum(
                p.get("realized_pnl") or p.get("realizedPnl") or 0
                for p in market_pos
            )

            # Get timestamps
            timestamps = [p.get("timestamp", 0) for p in market_pos if p.get("timestamp")]
            first_trade_ts = min(timestamps) if timestamps else None
            last_trade_ts = max(timestamps) if timestamps else None

            # Get event info from first position
            event_id = None
            event_slug = None
            market_title = None
            for p in market_pos:
                if not event_id:
                    event_id = p.get("event_id")
                if not event_slug:
                    event_slug = p.get("event_slug") or p.get("eventSlug")
                if not market_title:
                    market_title = p.get("title")

            # Generate unique ID
            market_stat_id = hashlib.sha256(f"{proxy_wallet}_{condition_id}".encode()).hexdigest()[:32]

            market_stats.append({
                "id": market_stat_id,
                "proxy_wallet": proxy_wallet,
                "event_id": event_id,
                "condition_id": condition_id,
                "wallet_volume": wallet_volume,
                "market_volume": 0.0,  # To be enriched with actual market volume
                "volume_share": 0.0,  # To be computed when market volume is available
                "n_trades": len(market_pos),
                "realized_pnl": realized_pnl,
                "first_trade_ts": first_trade_ts,
                "last_trade_ts": last_trade_ts,
                "market_title": market_title,
                "event_slug": event_slug,
                "computed_at": datetime.now(timezone.utc).isoformat()
            })

        logger.debug(f"Computed market stats for {proxy_wallet[:12]}...: {len(market_stats)} markets")
        return market_stats
