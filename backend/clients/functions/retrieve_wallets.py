"""Retrieve trades and wallets for events that still need raw data."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketDataAPI
from clients.wallet_tracker import WalletTracker

logger = logging.getLogger(__name__)


class RetrieveTradesWalletsCollector:
    """Fast path: fetch trades, persist wallets, mark events."""

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.data_api = PolymarketDataAPI()
        self.wallet_tracker = WalletTracker(api=self.data_api)
        self.trade_batch = 10_000
        self.max_requests = 8

    async def retrieve_wallets_from_events(self, max_events: int = 10, is_closed: bool = False) -> Dict[str, Any]:
        """Discover wallets for pending events in parallel."""
        logger.info(f"Starting wallet discovery for {max_events} events")
        start_time = datetime.now(timezone.utc)

        events = await self._get_events_needing_data_retrieval(max_events, is_closed)
        if not events:
            return {"events_processed": 0, "wallets_discovered": 0, "wallets_new": 0, "wallets_existing": 0, "duration_seconds": 0.0}

        async def process_event(event: Dict[str, Any]) -> tuple[str, Set[str]]:
            event_id = event["id"]
            wallets = await self._discover_wallets_only(event_id)
            return event_id, wallets

        event_results = await asyncio.gather(*[process_event(event) for event in events])

        all_unique_wallets = set()
        processed_events = []

        for event_id, wallets in event_results:
            all_unique_wallets.update(wallets)
            processed_events.append(event_id)

        logger.info(f"Processed {len(processed_events)} events, discovered {len(all_unique_wallets)} wallets")

        mark_tasks = [self._mark_event_retrieve_data(event_id, is_closed) for event_id in processed_events]
        await asyncio.gather(*mark_tasks)

        if all_unique_wallets:
            # Check which wallets already exist BEFORE creating wallet rows
            existing_wallets = await self._check_existing_wallets(list(all_unique_wallets))
            new_wallets = all_unique_wallets - existing_wallets

            # Only save NEW wallets that don't exist yet
            if new_wallets:
                wallet_rows = [{
                    "proxy_wallet": wallet_addr,
                    "enriched": False,
                    "total_trades": 0,
                    "total_volume": 0.0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                } for wallet_addr in new_wallets]

            await self._bulk_upsert("wallets", wallet_rows)
            logger.info(f"Saved {len(new_wallets)} new wallets to database")

            return {
                "events_processed": len(processed_events),
                "wallets_discovered": len(all_unique_wallets),
                "wallets_new": len(new_wallets),
                "wallets_existing": len(existing_wallets),
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
            }

        return {
            "events_processed": len(processed_events),
            "wallets_discovered": 0,
            "wallets_new": 0,
            "wallets_existing": 0,
            "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
        }

    async def _get_events_needing_data_retrieval(self, limit: int, is_closed: bool) -> List[Dict[str, Any]]:
        """Get events needing data retrieval."""
        table = "events_closed" if is_closed else "events"
        query = (self.db.supabase.table(table)
                .select("*")
                .eq("retrieve_data", False)
                .gt("market_count", 0)
                .gt("total_liquidity", 0))
        if not is_closed:
            query = query.eq("status", "active")
        query = query.order("total_liquidity", desc=True).order("market_count", desc=True).limit(limit)
        result = query.execute()
        return result.data or []

    async def _discover_wallets_only(self, event_id: str) -> Set[str]:
        """Discover unique wallets from event trades."""
        trades = await self._fetch_all_trades(event_id)
        if not trades: return set()

        wallets = set()
        for trade in trades:
            wallet = trade.get("proxyWallet") or trade.get("proxy_wallet")
            if wallet: wallets.add(wallet)

        logger.info(f"Event {event_id[:8]}: {len(trades)} trades, {len(wallets)} wallets")
        return wallets

    async def _fetch_all_trades(self, event_id: str) -> List[Dict[str, Any]]:
        """Fetch all trades for an event."""
        trades, pending, offset = [], set(), 0

        def schedule():
            nonlocal offset
            pending.add(asyncio.create_task(self.data_api.get_trades(event_id=[event_id], limit=self.trade_batch, offset=offset)))
            offset += self.trade_batch

        for _ in range(self.max_requests):
            schedule()

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                batch = task.result() or []
                if not batch:
                    pending.clear()
                    break
                trades.extend(batch)
                if len(batch) == self.trade_batch:
                    schedule()
        return trades

    async def _bulk_upsert(self, table: str, rows: List[Dict[str, Any]], chunk: int = 100) -> None:
        """Bulk upsert with small batches."""
        logger.info(f"Bulk upsert: {len(rows)} rows into {table}")

        for i, batch in enumerate(self._chunk(rows, chunk)):
                self.db.supabase.table(table).upsert(batch).execute()
                logger.info(f"Upserted batch {i+1} ({len(batch)} rows) into {table}")

    async def _mark_event_retrieve_data(self, event_id: str, is_closed: bool) -> None:
        """Mark event as retrieve_data=true."""
        await self.db.update_event_retrieve_data_status(event_id, True, is_closed)


    @staticmethod
    def _chunk(items: List[Any], size: int) -> List[Any]:
        for i in range(0, len(items), size):
            yield items[i:i + size]

    async def _check_existing_wallets(self, wallet_addresses: List[str]) -> Set[str]:
        """Check which wallets exist in database."""
        if not wallet_addresses: return set()
        existing = set()
        for batch in self._chunk(wallet_addresses, 200):
                result = (
                    self.db.supabase.table("wallets")
                    .select("proxy_wallet")
                    .in_("proxy_wallet", batch)
                    .execute()
                )
                existing.update(
                    row["proxy_wallet"]
                    for row in (result.data or [])
                    if row.get("proxy_wallet")
                )
        return existing



# Standalone function for easy import
async def retrieve_wallets_from_events(
    max_events: int = 100,  # Increased default to process more events at once
    is_closed: bool = False
) -> Dict[str, Any]:
    """Fast process: Retrieve wallets from events where retrieve_data=false (no trades storage)."""
    collector = RetrieveTradesWalletsCollector()
    return await collector.retrieve_wallets_from_events(max_events, is_closed)

